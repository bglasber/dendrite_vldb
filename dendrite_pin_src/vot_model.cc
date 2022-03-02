#include <unistd.h>
#include <cassert>
#include <unordered_set>
#include <cmath>
#include <fstream>
#include <iostream>

// Glorious gnu extension hacks
#include <ext/stdio_filebuf.h>

#include "vot_model.hh"
#include "im_trace.hh"
#include "trie.hh"

// All writes should be untracked.
#pragma GCC poison write

/*
 * alloc_new_metric_reservoir
 * alloc function for a metric reservoir.
 */
static MetricReservoir *alloc_new_metric_reservoir() {
    MetricReservoir *ptr = (MetricReservoir *) malloc( sizeof( MetricReservoir ) );
    ptr->next_slot = 0;
    ptr->count = 0;
    return ptr;
}

static size_t get_seq_length( const PriorEventSequence &seq ) {
    size_t len = seq.size();
    while( len >= 1 ) {
        if( seq[len-1] != -1 ) {
            break;
        }
        len--;
    };
    return len;
}

static int get_last_event_id( const PriorEventSequence &seq ) {
    int last_event_id = -1;
    for( int i = 0; i < ORDER_K; i++ ) {
        if( seq[i] != -1 ) {
            last_event_id = seq[i];
        }
    }
    assert( last_event_id != -1 );
    return last_event_id;
}

/*
 * process_new_data_for_dynamic_reservoir
 * Decides whether this thread should store new data point into
 * the given dynamic reservoir.
 */
void process_new_data_for_dynamic_reservoir(SequenceTransition *transition, uint64_t new_data, int metric_type_idx) {

    // Ignore if the new data is zero
    if(new_data == 0) {
        return;
    }

    // Alloc MetricReservoir when first time seeing this type of metric data
    if(transition->metrics_data[metric_type_idx] == NULL) {
        transition->metrics_data[metric_type_idx] = alloc_new_metric_reservoir();
    }

    int cur_reservoir_size = transition->metrics_data[metric_type_idx]->next_slot;

    if (cur_reservoir_size < RESERVOIR_SIZE) { 
        // Add new data point to the pool when current size is less than resevoir size
        transition->metrics_data[metric_type_idx]->value_pool[cur_reservoir_size] = new_data;
        transition->metrics_data[metric_type_idx]->next_slot++;
        transition->metrics_data[metric_type_idx]->count++;
        return;
    }

    float weight = (float) RESERVOIR_SIZE / (float) cur_reservoir_size;
    float f = (float) fastrand() / (float) RAND_RANGE;

    if( f <= weight ) {
        // Randomly replace one of the values
        int ind = fastrand() % RESERVOIR_SIZE;
        transition->metrics_data[metric_type_idx]->value_pool[ind]  = new_data;
    }

    // increase the count event
    transition->metrics_data[metric_type_idx]->count++;
}

/*
 * process_metrics_info
 * Update all metrics data reservoir of the given transition
 * Set every field in metrics_info back to zero
 */
void process_metrics_info(SequenceTransition *transition, double metrics_info[METRIC_INFO_SIZE]) {

    if(metrics_info == NULL) {
        return;
    }

    for (int i = 0; i < METRIC_INFO_SIZE; i++) {
        process_new_data_for_dynamic_reservoir(transition, metrics_info[i], i);
        metrics_info[i] = 0;
    }
}

/*
 * get_metric_count
 * Return the total number of data points collected for the
 * given metric type.
 */
int get_metric_count(SequenceTransition *transition, int metric_type_idx) {
    // return zero if can't find the key
    if(transition->metrics_data[metric_type_idx] == NULL) {
        return 0;
    }

    return transition->metrics_data[metric_type_idx]->count;
}

/*
 * Return if none of the full_seq_tr_probs differ from subseq_tr_prob by more than epsilon
 */
bool VariableOrderTransitionModel::can_reduce_fullseqs_to_subseq(
    double subseq_tr_prob,
    const std::vector<double> &fullseq_tr_probs,
    double epsilon
) {
    for( const double &fullseq_tr_prob : fullseq_tr_probs ) {
        double abs_tr_prob_diff = 0.0;
        if( fullseq_tr_prob > subseq_tr_prob ) {
            abs_tr_prob_diff = fullseq_tr_prob - subseq_tr_prob;
        } else {
            abs_tr_prob_diff = subseq_tr_prob - fullseq_tr_prob;
        }
        if( abs_tr_prob_diff > epsilon ) {
            return false;
        }
    }
    return true;
}

/*
 * Get all the events these sequences transition to. 
 */
std::unordered_set<int> VariableOrderTransitionModel::get_all_next_event_ids(
    const std::list<PriorEventSequence> &seqs
) {
    std::unordered_set<int> all_next_event_ids;
    for( const auto &seq_with_same_subseq : seqs ) {
        auto seq_search = prior_event_seq_table_.find( seq_with_same_subseq );
        assert( seq_search != prior_event_seq_table_.end() );
        for( auto tr_iterator : seq_search->second.transitions_ ) {
            all_next_event_ids.insert( tr_iterator.first );
        }
    }
    return all_next_event_ids;
}

// Chop off the first event, move everything in tmp to the left, then put -1s in the last position.
// AKA produce the next subsequence of this sequence
static void get_subseq( PriorEventSequence &dst, PriorEventSequence &src ) {
    unsigned dst_pos = 0;
    for( unsigned src_pos = 1; src_pos < ORDER_K; src_pos++ ) {
        dst[dst_pos] = src[src_pos];
        dst_pos++;
    }
    dst[dst_pos] = -1;
}

/*
 * Lookup the sequence and find what it transitions to.
 * If this sequence has already been reduced, return the reduced sequence.
 */
robin_hood::detail::Table<true, 80, PriorEventSequence, SequenceTransitions, robin_hood::hash<PriorEventSequence, void>, std::equal_to<PriorEventSequence > >::Iter<false> VariableOrderTransitionModel::find_seq_or_subseq(
    PriorEventSequence &prior_event_seq
) {

    // Fast path --- its in the map, and not reduced
    robin_hood::detail::Table<true, 80, PriorEventSequence, SequenceTransitions, robin_hood::hash<PriorEventSequence, void>, std::equal_to<PriorEventSequence > >::Iter<false> search =  prior_event_seq_table_.find( prior_event_seq );
    if( search != prior_event_seq_table_.end() ) {
        return search;
    }

    // Slow path, didn't immediately find it. Create copy and search for subsequences
    PriorEventSequence tmp = prior_event_seq;

    // There are K-1 subsequences we could have reduced to (by chopping off the front events).
    for( unsigned num_subseqs = 0; num_subseqs < ORDER_K - 1; num_subseqs++ ) {
        // Chop off the first event, move everything in tmp to the left, then put -1s in the last position.
        // AKA produce the next subsequence of this sequence
        get_subseq( tmp, tmp );

        search = prior_event_seq_table_.find( tmp );
        if( search != prior_event_seq_table_.end() ) {
            return search;
        }
    }
    return prior_event_seq_table_.end();
}

/*
 * Insert this sequence if not exists, and increment its transition count to next_event_id by one.
 */
std::pair<PriorEventSequence, uint64_t> VariableOrderTransitionModel::upsert_and_increment_transition(
    PriorEventSequence &prior_event_seq,
    int next_event_id,
    double metrics_info[METRIC_INFO_SIZE]
) {
    auto search = find_seq_or_subseq( prior_event_seq );
    if( search == prior_event_seq_table_.end() ) {
        SequenceTransitions seq_transitions;
        insert_seq_with_transitions( prior_event_seq, seq_transitions );

        //Try again
        return upsert_and_increment_transition( prior_event_seq, next_event_id, metrics_info );
    }

    auto &found_trs = search->second;
    uint64_t new_count = found_trs.increment_transition( next_event_id, metrics_info );

    return std::make_pair( search->first, new_count );
}

/*
 *  Insert a non-existing sequence into the transition table with the attached transitions.
 */
void VariableOrderTransitionModel::insert_seq_with_transitions(
    PriorEventSequence &seq,
    SequenceTransitions &transitions
) {
    // Put the Sequence into the table.

    prior_event_seq_table_.emplace( seq, transitions );

    // Since the transition does not exist, neither does its subsequence
    PriorEventSequence prior_event_subseq;
    {
        // Copy from 1->N of seq into 0->N-1 slots in prior_event_subseq
        size_t i = 1;
        for( ; i < get_seq_length( seq ); i++ ) {
           prior_event_subseq[i-1] = seq[i];
        }
        // OK, walk back one slot to convert to our last position in prior_event_subseq
        i--;

        //Fill the rest with -1.
        for( ; i < ORDER_K; i++ ) {
            prior_event_subseq[i] = -1;
        }
    }

    auto ins_iter = subsequence_index_.emplace( prior_event_subseq, std::list<PriorEventSequence>() );

    // Iterator->first is a pointer key-value pair. Second item is the list.
    // Push this PriorEventSequence on the list
    PriorEventSequence cons_seq = seq;
    ins_iter.first->second.push_back( cons_seq );

    // Add a reference to this sequence on the end index
    int last_event_id = get_last_event_id( seq );

    auto end_ind_iter = end_event_to_seq_index_.find( last_event_id );
    if( end_ind_iter == end_event_to_seq_index_.end() ) {
        end_event_to_seq_index_.emplace( last_event_id, std::list<PriorEventSequence>() );
        end_ind_iter = end_event_to_seq_index_.find( last_event_id );
    }
    end_ind_iter->second.push_back( cons_seq );
}

/*
 * Check if:
 * the last N-1 elements in their N elements are a match.
 */
bool VariableOrderTransitionModel::is_rev_matching_subseq(
    const PriorEventSequence &curr_seq,
    const PriorEventSequence &candidate_seq
) {
    auto rev_curr_seq_iter = curr_seq.rbegin();
    auto rev_other_seq_iter = candidate_seq.rbegin();
    size_t curr_seq_len = curr_seq.size();
    while( rev_curr_seq_iter != curr_seq.rend() and *rev_curr_seq_iter == -1 ) {
        rev_curr_seq_iter++;
        curr_seq_len--;
    }

    while( rev_other_seq_iter != candidate_seq.rend() and *rev_other_seq_iter == -1 ) {
        rev_other_seq_iter++;
    }


    for( size_t rev_offset = 0; rev_offset < curr_seq_len - 1; rev_offset++ ) {
        if( *rev_curr_seq_iter != *rev_other_seq_iter ) {
            return false;
        }
        rev_curr_seq_iter++;
        rev_other_seq_iter++;
    }

    return true;
}

/*
 * Do not allow us to reduce from (X,Y) to (Y) unless there are no other (W,X,Y) sequences.
 * returns true if (W,X,Y) exists. Avoids double reduction (and information loss) in one step.
 */
bool VariableOrderTransitionModel::reducing_would_skip_step( const PriorEventSequence &curr_seq ) const {
    
    
    int last_event_id = get_last_event_id( curr_seq );
    auto search = end_event_to_seq_index_.find( last_event_id );
    for( const PriorEventSequence &seq_ending_with_same_id : search->second ) {

        if( !is_rev_matching_subseq( curr_seq, seq_ending_with_same_id ) ) {
            continue;
        }

        if( get_seq_length( seq_ending_with_same_id ) > get_seq_length( curr_seq ) ) {
            return true;
        }
    }
    return false;
}

/*
 * If the subsequence of these lists has the same (within epsilon) transition probabilities
 * for each transition in all_next_event_ids, we can reduce. return true if so, false otherwise.
 */
bool VariableOrderTransitionModel::subseq_probs_differ_less_than_epsilon(
    const std::list<PriorEventSequence> &full_sequence_list,
    const std::unordered_set<int> all_next_event_ids
) {
    for( const int &next_event_id : all_next_event_ids ) {

        uint64_t subseq_occurence_count = 0;
        uint64_t subseq_this_tr_count = 0;

        std::vector<double> fullseq_tr_probs;
        
        for( const auto &seq_with_same_subseq : full_sequence_list ) {
            auto seq_search = prior_event_seq_table_.find( seq_with_same_subseq );
            assert( seq_search != prior_event_seq_table_.end() );

            /*
             * For each s_0,...,s_k, we need to get the times its moved to s_{k+1}
             * and the times its occurred.
             */
            uint64_t fullseq_occurence_count = 0;
            uint64_t fullseq_this_tr_count = 0;
            auto &trs = seq_search->second;
            for( auto tr_iterator : trs.transitions_ ) {
                uint64_t occurence_count = tr_iterator.second.transition_count_;
                fullseq_occurence_count += occurence_count;
                if( tr_iterator.first == next_event_id ) {
                    fullseq_this_tr_count += occurence_count;
                }
            }

            subseq_occurence_count += fullseq_occurence_count;
            subseq_this_tr_count += fullseq_this_tr_count;

            // Compute probability of s_0,...,s_k -> s_{k+1}
            assert( fullseq_occurence_count > 0 );
            double fullseq_tr_prob = ((double) fullseq_this_tr_count) / fullseq_occurence_count;
            assert( fullseq_tr_prob >= 0.0 and fullseq_tr_prob <= 1.0 );
            fullseq_tr_probs.push_back( fullseq_tr_prob );
        }

        // Compute probability of s_1,...,s_k -> s_{k+1}
        double subseq_tr_prob = ((double) subseq_this_tr_count) / subseq_occurence_count;
        assert( subseq_tr_prob >= 0.0 and subseq_tr_prob <= 1.0 );

        bool can_reduce = can_reduce_fullseqs_to_subseq( subseq_tr_prob, fullseq_tr_probs, epsilon_ );
        if( !can_reduce ) {
            return false;
        }
    }
    return true;
}

/*
 * Increment the transition count from prior_event_ids to next_event_id, inserting it if necessary.
 * Reduces prior_event_ids (or matching subsequence we track for it) by one if we can.
 */
void VariableOrderTransitionModel::increment_seq_transition_count( PriorEventSequence &prior_event_seq, int next_event_id, double metrics_info[METRIC_INFO_SIZE]) {

    std::pair<PriorEventSequence, uint64_t> seq_and_count = upsert_and_increment_transition( prior_event_seq, next_event_id, metrics_info);


    if( get_seq_length( seq_and_count.first ) <= 1 or seq_and_count.second < sample_reduction_threshold_ ) {
        // FIXME: a little conservative.
        // All that matters is that the total count is greater than sample_reduction_threshold_ for any
        // of my transitions.
        return;
    }

    if( seq_and_count.second % reduction_backoff_ != 0 ) {
        return;
    }

    /*
     * For the current k length sequence s_0, s_1, ... s_k -> s_{k+1} we are going to
     * try and reduce it to s_1, ... s_{k}-> s_{k+1}} if:
     * |P(s_{k+1}|s_0,...,s_k) - P(s_{k+1}|s_1,...,s_k| <= THETA \forall s_{k+1} and fixed 
     * s_0,...s_k.
     *
     * Step 1: Obtain s_0,...,s_k and s_1,...,s_k sequence transitions (s_{k+1}).
     * search->second holds s_{k+1} for s_0,...s_k. So look up subsequences.
     */

    // It is possible that we have already reduced the seq, so we need to get what the current seq is.
    const PriorEventSequence &curr_seq = seq_and_count.first;

    /*
     * To perform a reduction from (s_1,...,s_k) to (s_2,...,s_k) for fixed
     * s_2 ... s_k, and any s_1, we require that |P(s_{k+1}|s_1_...s_k) - 
     * P(s_{k+1}|s_2...s_k) | < epsilon for all s_1, s_{k+1}. 
     * Therefore, this reduction requries that there are no (s_0,s_1,...s_k) that we
     * have yet to reduce to (s_1,...,s_k) for any s_0, s_1. If there are, they haven't met
     * probability requriements to reduce yet.
     * If they can't reduce one step, we aren't going to try to reduce 2 steps. This avoids
     * a combinatorial search for subsets here. Theoretically it is possible to do this reduction,
     * but it is __lossy__ and expensive.
     *
     * So, if we are reducing from (s_1,...,s_k) to (s_2,...,s_k) we need to find all sequences that
     * end in (s_2,...,s_k) and make sure they are the same length as (s_1,...,s_k). We'll look up sequences
     * that end in s_k and then reverse check back. Making an index for all subsequence combos would be expensive,
     * but there probably aren't that many sequences that end in s_k.
     */
    if( reducing_would_skip_step( curr_seq ) ) {
        return;
    }
   
    PriorEventSequence prior_event_subseq;
    {
        int i = 0;
        for( auto copy_iter = curr_seq.begin()+1; copy_iter != curr_seq.end(); copy_iter++ ) {
            prior_event_subseq[i] = *copy_iter;
            i++;
        }
        for( ; i < ORDER_K; i++ ) {
            prior_event_subseq[i] = -1; // Tack -1 on the end.
        }
    }

    auto index_search = subsequence_index_.find( prior_event_subseq );
    assert( index_search != subsequence_index_.end() );

    // This is a list of all s_0,...,s_{k} sequences with *any* s_0.
    std::list<PriorEventSequence> &full_sequence_list = index_search->second;

    // If any transition that we would reduce down does not have enough transitions,
    // then don't bother.
    for( const auto &other_seq : full_sequence_list ) {
        auto search = prior_event_seq_table_.find( other_seq );
        auto &found_trs = search->second;
        for( const auto &seq_iter : found_trs.transitions_ ) {
            const SequenceTransition &seq = seq_iter.second;
            if( seq.transition_count_ < sample_reduction_threshold_ ) {
                return;
            }
        }
    }

    // Construct a list of set s_{k+1} to which any s_1,..,s_k subsequence transitions.
    std::unordered_set<int> all_next_event_ids = get_all_next_event_ids( full_sequence_list );

    // If we don't lose information by performing the reduction, do so.
    if( subseq_probs_differ_less_than_epsilon( full_sequence_list, all_next_event_ids ) ) {
        reduce_order_by_one( full_sequence_list, prior_event_subseq );
        (void) full_sequence_list;
    }
}

/*
 * Assign metrics reservoir after reduction to sequence transition
 */
void SequenceTransitions::assign_metrics_reservoir( int next_event_id, MetricReservoir* reservoirs[METRIC_INFO_SIZE]) {
    for( auto &entry : transitions_ ) {
        if( entry.first == next_event_id ) {
            for( int i = 0; i < METRIC_INFO_SIZE; i++ ) {
                entry.second.metrics_data[i] = reservoirs[i];
            }
            return;
        }
    }
    assert( false );
}

/*
 * Resample data from individual reservoir to the new reservoir
 */
void resample(MetricReservoir** new_reservoir, MetricReservoir* const* reservoir) {
    for (int i = 0; i < METRIC_INFO_SIZE; i++) {
        if (reservoir[i] == NULL) {
            continue;
        }
        // for each individual reservoir, we re-sample:
        // (current reservoir size / sum of all reservoirs sizes) * RESERVOIR_SIZE
        //  data points into the new reduced reservoir
        int size = std::ceil(reservoir[i]->count * RESERVOIR_SIZE / new_reservoir[i]->count);
        for(int j = 0; j < size; j++) {
            int ind = fastrand() % reservoir[i]->next_slot;
            if (new_reservoir[i]->next_slot < RESERVOIR_SIZE) {
                new_reservoir[i]->value_pool[new_reservoir[i]->next_slot++] = reservoir[i]->value_pool[ind];
            } else {
                int rand_idx = fastrand() % RESERVOIR_SIZE;
                new_reservoir[i]->value_pool[rand_idx] = new_reservoir[i]->value_pool[ind];
            }
        }
    }
}

/*
 * Reduce each sequence in full_sequence_list to prior_event_subseq.
 * OPTIMIZEME: A lot of the code in here is functionally similar to the check to see 
 * if we can reduce in the first place in increment_seq_transition_count.
 */
void VariableOrderTransitionModel::reduce_order_by_one(
    std::list<PriorEventSequence> &full_sequence_list,
    PriorEventSequence &prior_event_subseq
) {

    // Construct all the right transitions for the subseq
    SequenceTransitions subseq_transitions;

    // Store orignal reservoirs for each full sequence -> end_event tansition for re-sampling
    std::unordered_map<int, MetricReservoir**> resample_reservoirs;
    std::unordered_map<int, MetricReservoir**>::iterator reservoirs_search;


    // Pass 1: Construct maps from each next_event_id to an array of metric_reservoirs
    // Also figure out what the count of each of these metrics should be (sum them).
    for( const PriorEventSequence &seq : full_sequence_list ) {
        auto seq_search = prior_event_seq_table_.find( seq );
        assert( seq_search != prior_event_seq_table_.end() );
        for( const auto &seq_transition : seq_search->second.transitions_ ) {
            int next_event = seq_transition.first;
            reservoirs_search = resample_reservoirs.find( next_event );
            if( reservoirs_search == resample_reservoirs.end()) {
                MetricReservoir** new_reservoirs = (MetricReservoir**) malloc(METRIC_INFO_SIZE * sizeof(MetricReservoir*));
                for (int i = 0; i < METRIC_INFO_SIZE; i++) {
                    new_reservoirs[i] = NULL;
                }
                resample_reservoirs.emplace( next_event, new_reservoirs );
                reservoirs_search = resample_reservoirs.find( next_event );
            }
            for(int j = 0; j < METRIC_INFO_SIZE; j++) {
                if(seq_transition.second.metrics_data[j] != NULL) {
                    if(reservoirs_search->second[j] == NULL) {
                        reservoirs_search->second[j] = alloc_new_metric_reservoir();
                    }
                    reservoirs_search->second[j]->count += seq_transition.second.metrics_data[j]->count;
                }
            }
        }
    }

    // Pass 2: For each of the next_event_ids, take a sample of entries from their respective metrics.
    // Also, on this pass, sum up all the transition counts for the reduced subsequence
    for( const PriorEventSequence &seq : full_sequence_list ) {
        auto seq_search = prior_event_seq_table_.find( seq );
        assert( seq_search != prior_event_seq_table_.end() );
        
        auto &found_trs = seq_search->second;
        for( const auto &seq_transition : found_trs.transitions_ ) {
            int next_event = seq_transition.first;
            reservoirs_search = resample_reservoirs.find( next_event );
            assert( reservoirs_search != resample_reservoirs.end());
            // Resample new reservoir
            resample(reservoirs_search->second, seq_transition.second.metrics_data);
            subseq_transitions.increment_transition( seq_transition.first, seq_transition.second.transition_count_ , NULL /* No metrics yet */);
        }
        // Remove this full sequence, it will be subsumed by the subsequence
        prior_event_seq_table_.erase( seq );

        int last_event_id = get_last_event_id( seq );
        auto end_ind_iter = end_event_to_seq_index_.find( last_event_id );
        assert( end_ind_iter != end_event_to_seq_index_.end() );

#if !defined( NDEBUG )
        bool confirm_deleted = false;
#endif
        // OPTIMIZEME: This linear scan of the list isn't the greatest, but there's probably not too many sequences
        // ending in the same thing. If there are, this is something we can optimize.
        for(std::list<PriorEventSequence>::iterator list_iter = end_ind_iter->second.begin();
            list_iter != end_ind_iter->second.end();
            list_iter++
        ) {
            bool is_match = true;
            for( int i = 0; i < ORDER_K; i++ ) {
                if( (*list_iter)[i] != seq[i] ) {
                    is_match = false;
                    break;
                }
            }
            if( is_match ) {
                end_ind_iter->second.erase( list_iter );
#if !defined( NDEBUG )
                confirm_deleted = true;
#endif
                break;
            }
        }
#if !defined( NDEBUG )
        assert( confirm_deleted == true );
#endif
    }

    // Save each new reservoir into subseq_transitions
    for (reservoirs_search = resample_reservoirs.begin(); reservoirs_search != resample_reservoirs.end(); reservoirs_search++) {
        int end_event_id = reservoirs_search->first;
        subseq_transitions.assign_metrics_reservoir( end_event_id, reservoirs_search->second );
    }

    insert_seq_with_transitions( prior_event_subseq, subseq_transitions );

    // Remove the old subsequence from the index
    subsequence_index_.erase( prior_event_subseq );
}

static int write_seq_transition_to_buffer(
    const PriorEventSequence &seq,
    const SequenceTransition &transition,
    char *buff,
    int bufflen
) {
    int tmp_wrlen = 0;
    int buffer_offset = 0;
    int wrlen = 0;
    tmp_wrlen = snprintf( buff, bufflen, "(" );
    buffer_offset += tmp_wrlen;
    wrlen += tmp_wrlen;

    size_t seq_len = get_seq_length( seq );
    for( size_t i = 0; i < seq_len-1; i++ ) {
        tmp_wrlen = snprintf( buff+buffer_offset, bufflen-buffer_offset, "%d,", seq[i] );
        buffer_offset += tmp_wrlen;
        wrlen += tmp_wrlen;
    }
    wrlen += snprintf( buff+buffer_offset, bufflen-buffer_offset, "%d) -> %d", seq[seq_len-1], transition.next_event_id_ );
    return wrlen;
}

/*
 * Write a particular transition from seq -> transition to file.
 */
void VariableOrderTransitionModel::write_transition_to_file(
    int fd,
    const PriorEventSequence &seq,
    const SequenceTransition &transition
) const {
    char buff[512];
    int wrlen = write_seq_transition_to_buffer( seq, transition, buff, sizeof( buff ) );
    wrlen += snprintf( buff+wrlen, sizeof(buff)-wrlen, ": %ld\n", transition.transition_count_ );
    looped_write_to_file( fd, buff, wrlen );
}

/*
 * Write metrics info of a particular transition from seq -> transition -> metrics to file.
 */
void VariableOrderTransitionModel::write_transition_metrics_to_file(
    int fd,
    const PriorEventSequence &seq,
    const SequenceTransition &transition
) const {

    char buff[512];
    int wrlen = write_seq_transition_to_buffer( seq, transition, buff, sizeof( buff ) ); 
    wrlen += snprintf( buff+wrlen, sizeof(buff)-wrlen, ":\n" );
    looped_write_to_file( fd, buff, wrlen );

    // write metrics data
    for (int i = 0; i < METRIC_INFO_SIZE; i++ ) {
        switch (i) {
        case MALLOC_METRIC_IDX:
            looped_write_to_file( fd, "malloc: ", 8 );
            break;
        case FREE_METRIC_IDX:
            looped_write_to_file( fd, "free: ", 6 );
            break;
        case READ_METRIC_IDX:
            looped_write_to_file( fd, "read: ", 6 );
            break;
        case WRITE_METRIC_IDX:
            looped_write_to_file( fd, "write: ", 7 );
            break;
        case RECV_METRIC_IDX:
            looped_write_to_file( fd, "recv: ", 6 );
            break;
        case SEND_METRIC_IDX:
            looped_write_to_file( fd, "send: ", 6 );
            break;
        case TRANSITION_TIME_IDX:
            looped_write_to_file( fd, "time: ", 6 );
            break;
        }
        if (transition.metrics_data[i] == NULL) {
            looped_write_to_file( fd, "empty", 5 );
        } else {
            wrlen = 0;
            wrlen = snprintf( buff, sizeof( buff ),  "%ld;", transition.metrics_data[i]->count );
            looped_write_to_file( fd, buff, wrlen );
            for (int j = 0; j < transition.metrics_data[i]->next_slot; j++) {
                wrlen = snprintf( buff, sizeof( buff ), "%lf,", transition.metrics_data[i]->value_pool[j] );
                looped_write_to_file( fd, buff, wrlen );
            }
        }

        looped_write_to_file( fd, "\n", 1 );
    }
}

/*
 * Write the entire VariableOrderTransitionModel to disk (fd).
 */
void VariableOrderTransitionModel::write_to_file( int fd ) {
    
    char buff[128];
    memset( buff, '\0', sizeof( buff ) );
    int wrlen = snprintf( buff, sizeof(buff), "%f %ld %d\n", epsilon_, sample_reduction_threshold_, ORDER_K );
    looped_write_to_file( fd, buff, wrlen );

    for( auto &seq_iter : prior_event_seq_table_ ) {
        auto &trs = seq_iter.second;
        for( const auto &tr_iter : trs.transitions_ ) {
            const SequenceTransition &transition = tr_iter.second;
            write_transition_to_file( fd, seq_iter.first, transition );
        }
    }
}

/*
 * Write the all metrics data to disk (fd).
 */
void VariableOrderTransitionModel::write_metrics_to_file( int fd ) {
    for( const auto &seq_iter : prior_event_seq_table_ ) {
        auto &trs = seq_iter.second;
        for( const auto &tr_iter : trs.transitions_ ) {
            const SequenceTransition &transition = tr_iter.second;
            write_transition_metrics_to_file( fd, seq_iter.first, transition );
        }
    }
}

/*
 * Increment the transition to next_event_id by 1.
 */
uint64_t SequenceTransitions::increment_transition( int next_event_id , double metrics_info[METRIC_INFO_SIZE]) {
    return increment_transition( next_event_id, 1, metrics_info);
}

/*
 * Increment the transition to next_event_id by count. Insert if necessary. Update metrics reservoir.
 */
uint64_t SequenceTransitions::increment_transition( int next_event_id, uint64_t count, double metrics_info[METRIC_INFO_SIZE]) {
    auto search = std::find_if( transitions_.begin(), transitions_.end(),
        [&next_event_id](const std::pair<int,SequenceTransition> &entry) { return entry.first == next_event_id; } );
    if( search == transitions_.end() ) {
        SequenceTransition newSequenceTansition = SequenceTransition( next_event_id, count);
        process_metrics_info(&newSequenceTansition, metrics_info);
        transitions_.emplace_back( next_event_id, newSequenceTansition );
        return count;
    }
    uint64_t cur_count = search->second.transition_count_ += count;
    process_metrics_info(&search->second, metrics_info);
    return cur_count;
}
