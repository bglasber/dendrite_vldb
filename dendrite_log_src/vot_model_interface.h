#ifndef __VOT_MODEL_INTERFACE_H__
// C wrapper for VariableOrderTransitionModel so we can integrate it with PG/SQLite

#include "im_trace.h"
#ifdef __cplusplus
extern "C" {
#endif

struct VariableOrderTransitionModel;
typedef struct VariableOrderTransitionModel* VotModelHandle;

VotModelHandle create_vot_model( double epsilon, uint64_t sample_reduction_threshold, int reduction_backoff );
void destroy_vot_model( VotModelHandle vot_model );
void vot_increment_transition_count( VotModelHandle vot_model, LastKEventIds *last_k_events_info, int next_event_id, double metrics_info[ METRIC_INFO_SIZE ] );
void vot_write_to_file( VotModelHandle vot_model, int fd );
void vot_write_metrics_to_file( VotModelHandle vot_model, int fd );

#ifdef __cplusplus
}
#endif

#define __VOT_MODEL_INTERFACE_H__
#endif
