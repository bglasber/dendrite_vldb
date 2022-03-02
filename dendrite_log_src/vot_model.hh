#pragma once

#include <list>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <array>
#include "im_trace.h"
#include "robin_hood.h"

/*
 * MetricReservoir
 * Used to store different type of metric data (malloc, read etc.) to compute CDFs.
 */
struct MetricReservoir;

typedef struct MetricReservoir {
    int next_slot;
    uint64_t count;
    double value_pool[RESERVOIR_SIZE];
} MetricReservoir;

/*
 * fast_srand
 * Seed the fast random number generator with the given seed
 */
void fast_srand( int seed );

/*
 * fastrand
 * Returns one pseudo-random integer, output range [0-32767]
 */
int fastrand();

typedef std::array<int, ORDER_K> PriorEventSequence;

namespace std {
    template <>
    struct hash<PriorEventSequence> {
        std::size_t operator()( const PriorEventSequence &seq ) const noexcept {
            size_t cur_hash = 0;
            // The size here is always ORDER_K. But we know the "extra" elements for
            // subsequences are always -1.
            for( size_t i = 0; i < seq.size(); i++ ) {
                cur_hash = cur_hash ^ seq[i];
                //Rotate left shift
                cur_hash = (cur_hash << 7) | ( cur_hash >> (64-7));
            }
            return cur_hash;
        }
    };
}

typedef struct SequenceTransition {

    SequenceTransition( int next_event_id, uint64_t transition_count ) : next_event_id_( next_event_id ), transition_count_( transition_count ) {
        for(int i = 0; i < METRIC_INFO_SIZE; i++) {
            metrics_data[i] = NULL;
        }
    }

    int next_event_id_;
    uint64_t transition_count_;
    MetricReservoir* metrics_data[METRIC_INFO_SIZE];
} SequenceTransition;

class SequenceTransitions {
public:
    SequenceTransitions() { }
    SequenceTransitions( const SequenceTransitions &other ) : transitions_( other.transitions_ ) {
    }

    uint64_t increment_transition( int next_event_id, double metrics_info[METRIC_INFO_SIZE] );
    uint64_t increment_transition( int next_event_id, uint64_t count, double metrics_info[METRIC_INFO_SIZE] );
    void assign_metrics_reservoir( int next_event_id, MetricReservoir* reservoirs[METRIC_INFO_SIZE]);

    std::vector<std::pair<int,SequenceTransition>> transitions_;

};

template <class T>
inline void hash_combine(std::size_t& seed, T const& v)
{
    seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template<typename T, std::size_t N>
struct std::hash<std::array<T, N>>
{
    typedef std::array<T, N> argument_type;
    typedef std::size_t result_type;
    result_type operator()(argument_type const& in) const
    {
        size_t size = in.size();
        size_t seed = 0;
        for (size_t i = 0; i < size; i++)
            //Combine the hash of the current vector with the hashes of the previous ones
            hash_combine(seed, in[i]);
        return seed;
    }

};


/*
 * The root structure of a VariableOrderMarkovModel
 * Forward facing, root -> priors -> next_event_id
 * Has index to lookup priors-1 for reduction purposes.
 */

class VariableOrderTransitionModel {

public:
    VariableOrderTransitionModel( double epsilon, uint64_t sample_reduction_threshold, int reduction_backoff ) :
        epsilon_( epsilon ), sample_reduction_threshold_( sample_reduction_threshold ), reduction_backoff_( reduction_backoff ), prior_event_seq_table_( 1000 ) {}
    void increment_seq_transition_count( PriorEventSequence &prior_event_ids, int next_event_id, double metrics_info[METRIC_INFO_SIZE]);
    void write_to_file( int fd );
    void write_metrics_to_file( int fd );

    static bool can_reduce_fullseqs_to_subseq( double subseq_tr_prob, const std::vector<double> &fullseq_tr_probs, double epsilon );
    static bool is_rev_matching_subseq( const PriorEventSequence &curr_seq, const PriorEventSequence &candidate_seq );

protected:
    void insert_seq_with_transitions( PriorEventSequence &seq, SequenceTransitions &transitions );
    robin_hood::detail::Table<true, 80, PriorEventSequence, SequenceTransitions, robin_hood::hash<PriorEventSequence, void>, std::equal_to<PriorEventSequence> >::Iter<false> find_seq_or_subseq( PriorEventSequence &seq );
    std::pair<PriorEventSequence,uint64_t> upsert_and_increment_transition( PriorEventSequence &seq, int next_event_id, double metrics_info[METRIC_INFO_SIZE] );

    void write_transition_to_file( int fd, const PriorEventSequence &seq, const SequenceTransition &transition ) const;
    void write_transition_metrics_to_file( int fd, const PriorEventSequence &seq, const SequenceTransition &transition ) const;

    void reduce_order_by_one( std::list<PriorEventSequence> &full_sequence_list, PriorEventSequence &subseq );

    bool reducing_would_skip_step( const PriorEventSequence &curr_seq ) const;
    bool subseq_probs_differ_less_than_epsilon(
        const std::list<PriorEventSequence> &full_sequence_list,
        const std::unordered_set<int> all_next_event_ids
    );
    std::unordered_set<int> get_all_next_event_ids( const std::list<PriorEventSequence> &seqs );

    std::unordered_map<PriorEventSequence, std::list<PriorEventSequence>> subsequence_index_;
    std::unordered_map<int, std::list<PriorEventSequence>> end_event_to_seq_index_;
    double epsilon_;
    uint64_t sample_reduction_threshold_;
    int reduction_backoff_;
    robin_hood::unordered_flat_map<PriorEventSequence, SequenceTransitions> prior_event_seq_table_;
    
};
