#include "vot_model.hh"
#include "vot_model_interface.h"
#include <cassert>
#include <array>

// All writes should be untracked_write
#pragma GCC poison write

VotModelHandle create_vot_model( double epsilon, uint64_t sample_reduction_threshold, int reduction_backoff ) {
    return new VariableOrderTransitionModel( epsilon, sample_reduction_threshold, reduction_backoff );
}

void destroy_vot_model( VotModelHandle vot_model ) {
    delete vot_model;
}

void vot_increment_transition_count( VotModelHandle vot_model, LastKEventIds *last_k_events_info, int next_event_id, double metrics_info[METRIC_INFO_SIZE] ) {
    // Compute an array of last k event ids based on last_k_events_info in order
    std::array<int, ORDER_K> last_k_event_ids;
        
    int cur_idx = last_k_events_info->begin;
    for ( int i = 0; i < ORDER_K; i++ ) {
        last_k_event_ids[i] = last_k_events_info->circular_queue[ cur_idx % ORDER_K ];
        cur_idx++;
    }

    vot_model->increment_seq_transition_count( last_k_event_ids, next_event_id, metrics_info );
}

void vot_write_to_file( VotModelHandle vot_model, int fd ) {
    vot_model->write_to_file( fd );
}

void vot_write_metrics_to_file( VotModelHandle vot_model, int fd ) {
    vot_model->write_metrics_to_file( fd );
}
