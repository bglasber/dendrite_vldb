#pragma once
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <sys/types.h>


#define NEVENTS 1024
#define ORDER_K 3
#define METRIC_INFO_SIZE 7
#define RESERVOIR_SIZE 1000
#define MALLOC_METRIC_IDX 0
#define FREE_METRIC_IDX 1
#define READ_METRIC_IDX 2
#define WRITE_METRIC_IDX 3
#define RECV_METRIC_IDX 4
#define SEND_METRIC_IDX 5
#define TRANSITION_TIME_IDX 6

#define RAND_RANGE 32767

#if defined __cplusplus
    extern "C" {
#endif

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

/*
 * Reservoir
 * Sorted, linked list of reservoirs for event_ids.
 * Used to store transition times to compute CDFs.
 */
struct Reservoir;

typedef struct Reservoir {
    int event_id;
    int next_slot;
    double value_pool[RESERVOIR_SIZE];
    struct Reservoir *next;
} Reservoir;

/*
 * TransitionCount
 * Sorted, linked list of event transition counts
 */
struct TransitionCount;

/* TODO Right now, we can keep the transition counts between pairs of events. However,
 * in an order-k markov model, the previous k events are relevant for this transition.
 * Need to keep transition counts on this basis, and then collapse them once we have
 * enough information to gauge that the extra k states are not useful.
 */
typedef struct TransitionCount {
    int event_id;
    uint64_t count;
    struct TransitionCount *next;
} TransitionCount;

/*
 * EventRegInfo
 * Information about a given event_id.
 * How many times an event has happened, where it comes from, etc.
 */
typedef struct EventRegInfo {
    char                *filename;
    int                 line_number;
    uint64_t            count;
    int                 level;
} EventRegInfo;

double* get_metric_info();
void set_metric_info( int idx, double val );

/*
 * LastKEventIds
 * Record the ids of last K events.
 */
typedef struct LastKEventIds {
    int circular_queue[ORDER_K];
    int begin;
    int size;
} LastKEventIds;

/*
 * init_im_tracing()
 * Initialize in memory tracing. When called, initializes all of the memory allocated for 
 * this thread's in memory tracing components.
 */
void init_im_tracing( void );

/*
 * get_overflow
 * Determine if we are out of event slots.
 */
int get_overflow( void );

/*
 * get_nevents()
 * Return the number of events we can track, used as bounds on the tracking arrays 
 */
int get_nevents( void );

/*
 * dump_im_tracing()
 * Dump all in memory tracing information to per-thread files for offline analysis.
 */
void dump_im_tracing( void );

/*
 * record_event()
 * Record an event for the given log line
 */
void record_event( const char *fname, int line_number, int level );

/*
 * set_sampling_decision
 * Decides whether this thread ought the sample the next transition
 * for the given event_id. Sets should_record_transition_time
 */
bool set_sampling_decision( int event_id );

/*
 * get_event_index()
 * Determine the id of this log line
 * Linear probe on hash collision because deletion isn't a thing
 */
int get_event_index( const char *fname, int line_number, int level );

/*
 * get_event_reg_infos
 * Get the EventRegInfos for the current thread.
 */
EventRegInfo *get_event_reg_infos();

/*
 * get_last_event_id()
 * Return the eventID of the most recent event
 */
int get_last_event_id();

/*
 * get_event_hash
 * Hash the event using its __FILE__ and __LINE__ information
 */
uint64_t get_event_hash( const char *fname, int line_number );

void stop_libc_monitoring();
void start_libc_monitoring();
bool *get_should_monitor_libc_calls();
ssize_t untracked_write( int fd, const void *buff, size_t sz );
void looped_write_to_file( int fd, const char *buff, int wrlen );

#if defined __cplusplus
}
#endif
