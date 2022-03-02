#define _GNU_SOURCE

//FIXME organize
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <semaphore.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include "im_trace.h"
#include "packed_pointer.h"
#include "vot_model_interface.h"

// Writes from im_trace should be untracked and use untracked_write by default.
// To sidestep this (e.g. when implementing untracked_write) use blessed_write.
#define blessed_write write
#pragma GCC poison write

#define RAND_RANGE 32767

// The fastrand functions have a range of 0-32767. If reservoir size is larger, then
// samples are no longer uniformly random.
_Static_assert( RESERVOIR_SIZE < RAND_RANGE, "Reservoir size must be less than PRNG range" );
/* 
 * Per-thread content 
 * Contains event counts, transition counts, records that describe events, and some flag variables to indicate state.
 * Fixed number of events defined by NEVENTS to avoid an extra pointer indirection.
 * These matrices are quite sparse. If space is a concern, then we can revisit with sparse matrix libraries to reduce overhead
 */
static const uint64_t NANO = 1000 * 1000 * 1000;
static const float EPSILON = 0.05f;
static const float DELTA = 0.05f;
#define SAMPLE_REDUCTION_THRESHOLD ((uint64_t) (3 * log(2/DELTA) * (2 + EPSILON)  / (EPSILON * EPSILON)))

#define LOCK_SEM_NAME "/lock_sem"
static __thread sem_t *global_race_sem = NULL;
#define READY_SEM_NAME "/rdy_sem"
static __thread sem_t *global_ready_sem = NULL;

typedef struct ShmData {
    atomic_int dump_epoch;
} ShmData;

static __thread ShmData *shm_data = NULL;
static __thread int t_dump_epoch = 0;
static __thread bool is_owner_of_shm = false;
static __thread int shmid = -1;

static __thread uint64_t last_recorded_time = 0;
static __thread uint64_t last_recorded_time_vom = 0;
static __thread uint64_t g_seed;
static __thread bool     should_record_transition_time = false;
static __thread bool     should_monitor_libc_calls = true;

// The last event we recorded, for transition purposes
static __thread int last_event_ind = -1;

static __thread EventRegInfo *event_info = NULL;
static __thread EventRegInfo *prev_event_info = NULL;

static __thread VotModelHandle vot_model;
static __thread VotModelHandle prev_vot_model;
// temp array for current metrics statistics: mem_alloc, 
// mem_free, read_size, write_size, recv_size, send_size
static __thread double metrics_info[METRIC_INFO_SIZE];

double* get_metric_info() {
    return metrics_info;
}

void set_metric_info( int idx, double val) {
    metrics_info[idx] += val;
}

static const int reduce_backoff = 1000;
static __thread uint64_t record_event_hit_count = 0;
static __thread LastKEventIds last_k_events_info;

// If we ran out of event slots, set this to indicate a problem
static __thread int overflow = 0;

// The file descriptor that we should dump the trace to at the end
static __thread int out_fd = -1;

// The name of the file that we should dump the trace to at the end
static __thread char out_file_name[256];

// The file descriptor that we should dump the metrics data to at the end
static __thread int out_metrics_fd = -1;

// The name of the file that we should dump the metrics data to at the end
static __thread char out_metrics_file_name[256];

// Indicates if we re in the fork handler.
static __thread int in_fork_handler = 0;

/*
 * rotate_left
 * rotating left shift
 */
static inline uint64_t rotate_left( uint64_t val, int shift ) {
    return ( val << shift ) | ( val >> (64-shift) );
}

// Technically, this is ordered. We can short circuit.
#define find_relevant_list_entry( base_ptr, event_id ) \
    while( base_ptr != NULL ) { \
        if( base_ptr->event_id == event_id ) { \
            break; \
        } \
        base_ptr = base_ptr->next; \
    } \
    (void) base_ptr

/*
 * init_last_k_events_info
 * set initial size, next_slot etc. for last_k_events_info
 */
static void init_last_k_events_info() {
    last_k_events_info.size = 0;
    last_k_events_info.begin = 0;
}

EventRegInfo *get_event_reg_infos() {
    return event_info;
}

/*
 * fast_srand
 * Seed the fast random number generator with the given seed
 */
void fast_srand( int seed ) {
    g_seed = seed;
}

/*
 * fastrand
 * Returns one pseudo-random integer, output range [0-32767]
 */
int fastrand() {
    g_seed = (214013*g_seed+2531011);
    return (g_seed>>16)&0x7FFF;
}

/*
 * looped_write_to_file
 * Given a file descriptor and a buffer, write wrlen bytes from the buffer
 * to the fd, looping as necessary until it is done.
 */
void looped_write_to_file( int fd, const char *buff, int wrlen ) {
    int wr_thus_far = 0;
    if( wrlen > 0 ) { 
        int write_ret = 0; 
        while( wr_thus_far < wrlen ) {
            write_ret = untracked_write( fd, buff+wr_thus_far, wrlen-wr_thus_far );
            if( write_ret == -1 ) {
                printf( "Could not write to file. Got error code: %d\n", errno );
                return;
            }
            wr_thus_far += write_ret;
        }
    }
}

/* normalize()
 * Take the ticks value and normalize by 10^4
 */
static inline double normalize( uint64_t ticks ) {
    return ((double)ticks) / (double) 10000.0;
}


void im_fork();

/* get_overflow()
 * Determine if we are out of event slots.
 */
int get_overflow() {
    return overflow;
}

/*
 * get_nevents()
 * Return the number of events we can track, used as bounds on the tracking arrays 
 */
int get_nevents() {
    return NEVENTS;
}

EventRegInfo *alloc_event_info_array() {
    EventRegInfo *ptr = (EventRegInfo *) malloc( sizeof(EventRegInfo) * NEVENTS );

    // Zero all of the memory for this thread.
    memset( out_file_name, '\0', sizeof(out_file_name) );
    for (int i = 0; i < NEVENTS; i++ ) {
        ptr[i].filename = NULL;
        ptr[i].line_number = 0;
        ptr[i].count = 0;
        ptr[i].level = 0;
    }

    return ptr;
}

struct sock_control_thread_args {
    int sockfd;
    struct sockaddr_in address;
    ShmData *shm_data;
};

void set_global_dump() {
    // This can be null if another thread in our process has decided we areclosing down. This is possible
    // at the end of the run, or if a thread calls exit() or something like that... Should this be a standalone process?
    if( shm_data != NULL ) {
        shm_data->dump_epoch++;
    }
}

static void *sock_control_thread( void *args_in ) {
    struct sock_control_thread_args *args = (struct sock_control_thread_args *) args_in;
    unsigned int addrlen = sizeof(args->address);
    assert( args->shm_data != NULL );
    assert( shm_data == NULL );
    shm_data = args->shm_data;

    int new_sock = accept( args->sockfd, (struct sockaddr *) &(args->address), &addrlen );
    if( new_sock < 0 ) {
        printf( "Could not accept connection, errcode: %d\n", errno );
        free( args );
        return NULL;
    }

    char buff[32];

    for( ;; ) {

        int rc = recv( new_sock, buff, 1, 0 );
        if( rc < 0 ) {
            printf( "Could not read from socket! errcode: %d\n", errno );
            free( args );
            return NULL;
        }
        if( rc == 0 ) {
            printf( "Received empty message, socket closed!\n" );
            close( new_sock );
            free( args );
            return NULL;
        }

        if( buff[0] == 'D' ) {
            // dump and start new epoch
            set_global_dump();
        } else if( buff[0] == 'I' ) {
            // Internal engine command
            uint32_t func_name_sz;
            rc = recv( new_sock, &func_name_sz, 4, 0 );
            assert( rc == 4 );
            rc = recv( new_sock, buff, func_name_sz, 0 );
            assert( rc == func_name_sz );
            buff[ func_name_sz ] = '\0';
            printf( "Told to execute function: %s", buff );
        } else {
            printf( "Received unknown command on socket!\n" );
        }

        // ACK
        buff[0] = 'A';

        rc = send( new_sock, buff, 1, 0 );
        assert( rc > 0 );
    }

    free( args );
    return NULL;
}

static void spawn_control_socket() {
    int opt = 1;

    struct sock_control_thread_args *args = malloc( sizeof(struct sock_control_thread_args) );
    memset( args, '\0', sizeof( struct sock_control_thread_args ) );
    
    if( (args->sockfd = socket( AF_INET, SOCK_STREAM, 0 )) < 0 ) {
        printf( "Could not create control socket!\n" );
        return;
    }

    if( (setsockopt( args->sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt) )) < 0 ) {
        printf( "Could not set socket options on control socket!\n" );
        return;
    }

    args->address.sin_family = AF_INET;
    args->address.sin_addr.s_addr = INADDR_ANY;
    args->address.sin_port = htons( 9999 );

    if( (bind( args->sockfd, (struct sockaddr *) &(args->address), sizeof( args->address ) )) < 0 ) {
        printf( "Could not bind socket to port: %d!\n", errno );
        return;
    }

    // This stuff should be in a thread and a loop
    if( (listen( args->sockfd, 3 ) ) < 0 ) {
        printf( "Could not listen to socket: %d!\n", errno );
    }

    // Set up shared memory structure so that the control thread can communicate with everyone, even if they fork.

    key_t shm_key = ftok( "tmpfile", 1 );
    shmid = shmget( shm_key, sizeof( ShmData ), 0644 | IPC_CREAT );
    if( shmid == -1 ) {
        printf( "Could not get shared memory segment, errno=%d\n", errno );
        assert( false );
    }
    shm_data = shmat( shmid, NULL, 0 );
    if( (void *) shm_data == (void *) -1 ) {
        // failed
        printf( "Could not attach to shared memory segment, errno=%d.\n", errno );
        assert( false );
    }

    memset( shm_data, '\0', sizeof( ShmData ) );
    shm_data->dump_epoch = 0;
    args->shm_data = shm_data;
   
    pthread_t thread1;
    pthread_create( &thread1, NULL, sock_control_thread, (void *) args );
}

/*
 * init_im_tracing()
 * Initialize in memory tracing. When called, initializes all of the memory allocated for 
 * this thread's in memory tracing components.
 */
void init_im_tracing() {

#if !defined( UNIT_TEST )
    // Map the global mutex by name, we should create it if it doesn't exist or use the one that is already here.
    if( global_race_sem == NULL ) {
        if( (global_race_sem = sem_open( LOCK_SEM_NAME, O_CREAT, 0600, 1 )) == SEM_FAILED ) {
            printf( "Could not map global race sem!\n" );
            return;
        }
    }

    if( global_ready_sem == NULL ) {
        if( (global_ready_sem = sem_open( READY_SEM_NAME, O_CREAT, 0600, 0 )) == SEM_FAILED ) {
            printf( "Could not map global ready sem!\n" );
            return;
        }
    }

    // Acquire the global mutex, only one person will get it
    for( ;; ) {
        int rc = sem_trywait( global_race_sem );
        if( rc == 0 ) {
            is_owner_of_shm = true;
            spawn_control_socket();
            // Post a whole bunch of times to let other people through.
            // Yes this is basically a condition variable
            // Don't think there are named condition variables though?
            for( int i = 0; i < 100; i++ ) {
                rc = sem_post( global_ready_sem );
                if( rc == -1 ) {
                    printf( "Could not post to sem: %d\n", errno );
                }
                assert( rc == 0 );
            }
            // Get out
            break;
        } else {
            // interrupted, go around
            if( errno == EINTR ) {
                continue;
            }

            // I need these guys to stop until a shared memory buffer is created.
            // I could have them shmget, but I don't want them to O_CREAT
            // and even if they don't O_CREAT, I want them to wait until the structure is well_formed.
            assert( rc == -1 );
            assert( errno == EAGAIN );

            // I need a second semaphore to signal that the shm segment can be mapped and used.
            // I can't use mmap because we might have already forked. Need a global thing for processes to map.
            // Second semaphore will wait and then unlock
            // Only thing in the shm is ShmData.

            for( ;; ) {
                rc = sem_wait( global_ready_sem );
                if( rc == -1 ) {
                    // interrupted, go around
                    if( errno == EINTR ) {
                        continue;
                    } else {
                        printf( "Could not wait for ready semaphore: %d\n", errno );
                        assert( false );
                        break;
                    }
                }

                // Let everyone else go past, we're ready!
                sem_post( global_ready_sem );

                // We need to be somewhat careful here because we may be init'd by a fork
                // after our parent already initialized this stuff. In that case, we'll get
                // shm_data from our parents.

                key_t shm_key = ftok( "tmpfile", 1 );
                if( shmid == -1 ) {
                    shmid = shmget( shm_key, sizeof( ShmData ), 0644 /* NO CREATE */ );
                    if( shmid == -1 ) {
                        printf( "Could not shmget in waiting process, errno=%d\n", errno );
                        assert( false );
                        break;
                    }
                }
                if( shm_data == NULL ) {
                    shm_data = shmat( shmid, NULL, 0 );
                    if( (void *) shm_data == (void *) -1 ) {
                        //failed
                        printf( "Could not shmat in waiting process, errno=%d\n", errno );
                        assert( false );
                        break;
                    }
                }
                break;
            }
            // Get out
            break;
        }
    }

    // Determine cur_epoch:
    assert( shm_data != NULL );
    t_dump_epoch = shm_data->dump_epoch;
#else
    t_dump_epoch = 0;
#endif

    event_info = alloc_event_info_array(); 
    // Zero all of the memory for this thread.
    memset( out_file_name, '\0', sizeof(out_file_name) );
    memset( out_metrics_file_name, '\0', sizeof(out_metrics_file_name) );

    for( int i = 0; i < NEVENTS; i++ ) {
        event_info[i].filename = NULL;
        event_info[i].line_number = 0;
        event_info[i].count = 0;
        event_info[i].level= 0;
    }

    // init metrics_info
    for( int i = 0; i < METRIC_INFO_SIZE; i++ ) {
        metrics_info[i] = 0;
    }

    // Initialize last_k_events_info
    init_last_k_events_info();

    // Initialize vot_model
    vot_model = create_vot_model( EPSILON, SAMPLE_REDUCTION_THRESHOLD, reduce_backoff );

    // Initialize variables so we know we are just starting up.
    last_event_ind = -1;

    // We haven't overflowed yet.
    overflow = 0;

    // Figure out our process and thread_id
    pid_t my_pid = getpid();
    pid_t my_tid = syscall( __NR_gettid );

    // Seed the random number generator
    fast_srand( my_pid * my_tid );

    /*
     * Now we want to set up a file so we can write to it when dump_im_tracing is called.
     * It could be the case that we already have an fd, and the caller executed this function to wipe
     * our memory (as done in the unit tests). If that is the case, don't bother opening a new
     * file.
     */

#if !defined( UNIT_TEST )
    if( out_fd == -1 ) {
        /*
         * We are going to add extensions in case this pid/tid pair already has an open file. This
         * should not happen, but can if there are residual files from previous processes kicking around
         * in /tmp and we looped around on pids.
         */
        for( int extension_offset = 0; ; extension_offset++ ) {
            memset( out_file_name, '\0', sizeof(out_file_name) );
            snprintf( out_file_name, sizeof(out_file_name), "/hdd1/dendrite_data/%d.%d.%d.im.out", my_pid, my_tid, extension_offset );
            out_fd = open( out_file_name, O_CREAT | O_WRONLY | O_TRUNC | O_EXCL, S_IRUSR | S_IWUSR );

            // If we got a valid return code, out_fd > 0.
            if( out_fd >= 0 ) {
                break;
            }

            /* If we received an error from open that is NOT EEXIST, it means something really
             * bad happened and we can't trace the file. Unfortunately, there isn't much we can
             * do about this problem. It seems overkill to crash a process because we couldn't trace
             * it, but writing to stdout/stderr may not work either because they may be closed or
             * redirected. Let's just print to console and hope the user notices...
             */
            if( out_fd == -1 && errno != EEXIST ) {
                printf( "WARNING: could not open file!\n" );
                break;
            }
            // EEXIST --- file already exists for PID/TID pair, bump the extension and try again.
        }
        // out_fd >= 0 or out_fd = -1 with some error.

    }

    if( out_metrics_fd == -1 ) {
        /*
         * We are going to add extensions in case this pid/tid pair already has an open file. This
         * should not happen, but can if there are residual files from previous processes kicking around
         * in /tmp and we looped around on pids.
         */
        for( int extension_offset = 0; ; extension_offset++ ) {
            memset( out_metrics_file_name, '\0', sizeof(out_metrics_file_name) );
            snprintf( out_metrics_file_name, sizeof(out_metrics_file_name), "/hdd1/dendrite_data/metrics/%d.%d.%d.im.metrics.out", my_pid, my_tid, extension_offset );
            out_metrics_fd = open( out_metrics_file_name, O_CREAT | O_WRONLY | O_TRUNC | O_EXCL, S_IRUSR | S_IWUSR );

            if( out_metrics_fd >= 0 ) {
                break;
            }

            /* If we received an error from open that is NOT EEXIST, it means something really
             * bad happened and we can't trace the file. Unfortunately, there isn't much we can
             * do about this problem. It seems overkill to crash a process because we couldn't trace
             * it, but writing to stdout/stderr may not work either because they may be closed or
             * redirected. Let's just print to console and hope the user notices...
             */
            if( out_metrics_fd == -1 && errno != EEXIST ) {
                printf( "WARNING: could not open file!\n" );
                break;
            }
            // EEXIST --- file already exists for PID/TID pair, bump the extension and try again.
        }
    }


    /*
     * Some processes fork to create clones of themselves, and we will want to trace all of
     * these successfully (e.g. PostgreSQL). If we fork, we will wipe the child's memory and
     * reinitialize its tracing module. We set up a fork handler to detect this.
     *
     * Setting up a fork_handler while in a fork handler results in an infinite loop, so avoid that.
     */
    if( !in_fork_handler ) {
        pthread_atfork( NULL, NULL, im_fork );
    }
#else 
    out_fd = 1;
#endif
}

/* 
 * im_fork
 * If we are forked, then the child process should reset their memory and get a fd to
 * dump its tracing to.
 */
void im_fork() {
    out_fd = -1;
    out_metrics_fd = -1;
    in_fork_handler = 1; // We are in the fork handler, don't set up another fork handler!
    is_owner_of_shm = false; // We forked, so we aren't the leader of shm any more.
    // FIXME do not remap semaphores/shm on fork, we inherit the address space
    init_im_tracing();
}

/*
 * get_last_event_id()
 * Return the eventID of the most recent event
 */
int get_last_event_id() {
    return last_event_ind;
}


/*
 * get_event_hash
 * Hash the event using its (pathless)  __FILE__ and __LINE__ information
 */
uint64_t get_event_hash( const char *pathless_fname, int line_number ) {
    assert( strrchr( pathless_fname, '/' ) == NULL );
    uint64_t len = strlen( pathless_fname );
    uint64_t cur_pos = 0;
    uint64_t cur_hash = 0;

    /*
     * If there are more than 8 bytes left in the file name, interpret the bytes
     * as a uint64_t and hash it in.
     */
    while( len - cur_pos > 8 ) {
        cur_hash = cur_hash ^ (* (uint64_t *) (pathless_fname + cur_pos));
        cur_hash = rotate_left( cur_hash, 7 );
        cur_pos += 8;
    }

    /*
     * If there are more than 4 bytes left in the file name, interpret the bytes
     * as a uint32_t and hash it in.
     */
    if( len - cur_pos > 4 ) {
        cur_hash = cur_hash ^ ( * (uint32_t *) (pathless_fname + cur_pos ));
        cur_hash = rotate_left( cur_hash, 7 );
        cur_pos += 4;
    }

    /*
     * Create a uint32_t from the remaining bytes in the filename by shifting in
     * the bits, and then hash it in.
     */
    uint32_t interim = 0;
    while( len - cur_pos > 0 ) {
        uint32_t val = (uint32_t) *(pathless_fname + cur_pos);
        interim = interim | val;
        interim = interim << 8;
        cur_pos++;
    }
    cur_hash = cur_hash ^ interim;
    cur_hash = rotate_left( cur_hash, 7 );
    cur_hash ^= line_number;
    return cur_hash;
}

/*
 * is_hash_entry_for
 * Check if the provided hash is the entry for the log event corresponding to the fname and
 * line number.
 */
static bool is_hash_entry_for( int hash, const char *pathless_fname, int line_number ) {
    assert( strrchr( pathless_fname, '/' ) == NULL );
    return event_info[ hash ].line_number == line_number &&
        //Pretty sure we can disable the strcmp because the fname ptr should be constant for a given logline
        //TODO: Ensure that log behaviour looks about the same before/after strcmp disable on PG
        ( (void *) event_info[ hash ].filename == (void *) pathless_fname || strcmp( event_info[ hash ].filename, pathless_fname ) == 0  );
}

/*
 * allocate this slot in the event info array
 * set up transition array and CDF ptr
 */
static void alloc_slot( int slot, const char *pathless_fname, int line_number, int level ) {
    assert( strrchr( pathless_fname, '/' ) == NULL );

    event_info[ slot ].filename = (char *) pathless_fname;
    event_info[ slot ].line_number = line_number;
    event_info[ slot ].level = level;

}

/*
 * get_event_index()
 * Determine the id of this log line
 * Linear probe on hash collision because deletion isn't a thing
 */
int get_event_index( const char *pathless_fname, int line_number, int level ) {
    assert( strrchr( pathless_fname, '/' ) == NULL );

    uint64_t pre_hash_id;
    int hash;
    int orig_ind_pos;

    pre_hash_id = get_event_hash( pathless_fname, line_number );
    // We need to jumble the bits around so we don't mod by line number
    // every time
    hash = pre_hash_id % NEVENTS;

    // Check hash slot
    if( event_info[ hash ].filename == NULL ) {
        // No one here, use this slot
        alloc_slot( hash, (char *) pathless_fname, line_number, level );
        return hash;
    }

    // Something is in the slot, check if we have a prior entry
    if( is_hash_entry_for( hash, pathless_fname, line_number ) ) {
        return hash;
    }

    // We have a hash collision, time to probe...
    orig_ind_pos = hash;

    while( ( hash = (hash + 1) % NEVENTS ) != orig_ind_pos ) {
        if( event_info[ hash ].filename == NULL ) {
            // Found an empty slot
            alloc_slot( hash,  pathless_fname, line_number, level );
            return hash;
        } else if( is_hash_entry_for( hash, pathless_fname, line_number ) ) {
            // Found our slot
            return hash;
        }
        //Try next slot
    }

    //Out of slots, fail
    return -1;
}

uint64_t get_current_nsec() {
    int ret = 0;
    struct timespec ts;
    ret = clock_gettime( CLOCK_REALTIME, &ts );
    assert( ret == 0 );
    (void) ret; // suppress unused warning
    uint64_t now = ts.tv_sec * NANO + ts.tv_nsec;
    return now;
}

/*
 * set_sampling_decision
 * Decides whether this thread ought the sample the next transition
 * for the given event_id. Sets should_record_transition_time
 */
bool set_sampling_decision( int event_id ) {

    uint64_t event_count = event_info[ event_id ].count;
    if( event_count == 0 ) {
        should_record_transition_time = true;
        return should_record_transition_time;
    }

    // Goes from RESERVOIR_SIZE -> 0.
    // Represents the probability of sampling
    float weight = (float) RESERVOIR_SIZE / (float) event_count;

    // Clamp fastrand() range to [0,1]
    float f = (float) fastrand() / (float) RAND_RANGE;

    // If f <= weight, then record this transition.
    if( f <= weight ) {
        should_record_transition_time = true;
        return should_record_transition_time;
    }

    should_record_transition_time = false;
    return should_record_transition_time;
}

static const char *remove_path_from_fname( const char *fname ) {
    /*const char *pos = strrchr( fname, '/' );
    if( pos != NULL ) {
        const char *pathless_fname = pos + 1; //skip the '/' char 
        return pathless_fname;
    }*/
    return fname;
}

/*
 * update_vo_transition_model
 * record the current event sequences to last_k_events_info and vot_model
 * record the current metrics statistics into the vot model
 */
static void update_vo_transition_model( int event_id ) {

    assert( last_event_ind != -1 );

    // Record the last event in last_k_events_info, increment the size
    last_k_events_info.circular_queue[ ( last_k_events_info.begin + last_k_events_info.size ) % ORDER_K] = last_event_ind;
    last_k_events_info.size++;

    if( last_k_events_info.size > ORDER_K ) {
        last_k_events_info.begin = (last_k_events_info.begin+1) % ORDER_K;
        last_k_events_info.size--;
        assert( last_k_events_info.size == ORDER_K );
    }
    
    if( last_k_events_info.size == ORDER_K ) {

        // Sanity Test
        // Ensure we know about each event!
        for( int last_k_pos = 0; last_k_pos < ORDER_K; last_k_pos++ ) {
            int last_k_offset = ( last_k_pos + last_k_events_info.begin ) % ORDER_K;
            int this_event_id = last_k_events_info.circular_queue[ last_k_offset ];
            assert( event_info[ this_event_id ].count > 0 );
            (void) this_event_id;
        }

        bool we_are_recording_metrics = should_record_transition_time;
        if( we_are_recording_metrics ) {
            // record transition time
            uint64_t cur_time = get_current_nsec();
            uint64_t last_time = last_recorded_time_vom;
            uint64_t elapsed_nsec = 0;

            if( last_time <= cur_time ) {
                elapsed_nsec = (cur_time - last_time);
            }

            double norm_elapsed_nsec = normalize( elapsed_nsec );
            last_recorded_time_vom = cur_time;
            metrics_info[TRANSITION_TIME_IDX] = norm_elapsed_nsec;
        }

        // OK, actually update things.
        vot_increment_transition_count( vot_model, &last_k_events_info, event_id, metrics_info );

        // Wipe metrics.
        for( int i = 0; i < METRIC_INFO_SIZE; i++ ) { 
            metrics_info[i] = 0;
        }
    }
}

/*
 * record_event()
 * Record an event for the given log line
 */
void record_event( const char *fname, int line_number, int level ) {
    int event_id;
    char buff[128];
    int buff_len;
    int wr_thus_far;
    int wr;

    // It could be the case that the fname we are using has the fullpath, which we don't want
    const char *pathless_fname = remove_path_from_fname( fname );

    if( out_fd == -1 ) {
        //If we aren't initialized, then first do that.
        init_im_tracing();
    }

    event_id = get_event_index( pathless_fname, line_number, level );

    if( event_id == -1 ) {
        //Disaster, we are going to lose this event type
        //Write this out to the file, should break parsers and warn us
        overflow = 1;
        memset( buff, '\0', sizeof( buff ) );
        buff_len = snprintf( buff, 128, "ERROR: Lost event type: %s:%d!\n", pathless_fname, line_number );
        wr_thus_far = 0;
        while( wr_thus_far < buff_len ) {
            wr = untracked_write( out_fd, buff+wr_thus_far, buff_len-wr_thus_far );
            wr_thus_far += wr;
        }
        return;
    }

    event_info[ event_id ].count += 1;

    uint64_t cur_time = 0;

    /* 
     * If should_record_transition_time is set, then we know that we are sampling *this* transition time.
     * therefore, we will have set the last_recorded_time. Get the current time, subtract the last recorded
     * time from it and obtain the transition time. Update the last_recorded time to what we obtained in case
     * we want to also time the next transition.
     *
     * Note that if the should_record_transition_time is *not* set, we may still wish to record the
     * elapsed time of the next transition. To do so, we need to current time (e.g. the time of this event).
     * We'll go back and get that later based on whether we decide to sample it (set_sampling_decision).
     */
    if( last_event_ind != -1 ) {
        update_vo_transition_model( event_id );
    }

    /*
     * N.B. We would like to sample transition times based on how often that transition occurs.
     * If we haven't observed a transition's time very often, we want more samples from it.
     * However, we can't know a priori what the upcoming transition is --- we know only what
     * the current event is. Therefore, we use the number of times we've seen the current event
     * to choose the sampling rate, assuming all transitions are equally likely. Clearly, this
     * isn't ideal. However, deciding to keep or throw away the sample later doesn't save much
     * overhead because we are already making a syscall for one timer here. Might as well make the
     * other at the end and subtract them if we are willing to tolerate that overhead.
     */

    bool we_recorded_last_transition_time = should_record_transition_time;

    // Sampling decision applies for the next transition.
    bool should_record_next_transition_time = set_sampling_decision( event_id );

    // If we should record *this* transition, but not the previous, then we need the current time.
    if( should_record_next_transition_time && !we_recorded_last_transition_time ) {
        cur_time = get_current_nsec();
        last_recorded_time = cur_time;
    // If we should record now (and we did record before, then save the current timer)
    } else if( should_record_next_transition_time )  {
        // Store whatever time we recorded above
        last_recorded_time = cur_time;
    }
    // Otherwise, we aren't sampling the next transition

    if( shm_data != NULL ) {
        if( shm_data->dump_epoch > t_dump_epoch ) {
    
            dump_im_tracing();
            if( prev_event_info != NULL ) {
                free( prev_event_info );
            }
            prev_event_info = event_info;
            event_info = alloc_event_info_array();

            if( prev_vot_model != NULL ) {
                destroy_vot_model( prev_vot_model );
            }
            prev_vot_model = vot_model;
            vot_model = create_vot_model( EPSILON, SAMPLE_REDUCTION_THRESHOLD, reduce_backoff );

            //Reset last K events...
            init_last_k_events_info();

            // There was no previous last_event_ind
            last_event_ind = -1;
            return;
        }
    }

    // Update last event
    last_event_ind = event_id;
    record_event_hit_count++;

}

void write_out_tracing() {
    char buff[512];
    int wrlen;

    for( int i = 0; i < NEVENTS; i++ ) {
        if( event_info[i].filename != NULL ) {
            memset( buff, '\0', sizeof( buff ) );
            wrlen = snprintf( buff, 512, "%s:%d = %d, %ld, %d\n", event_info[i].filename, event_info[i].line_number, i, event_info[i].count, event_info[i].level);
            looped_write_to_file( out_fd, buff, wrlen );
        }
    }

    vot_write_to_file( vot_model, out_fd );
    vot_write_metrics_to_file( vot_model, out_metrics_fd );

    buff[0] = '.';
    int epoch_name = t_dump_epoch; // Enables attribution to previous epochs if I just woke up.
    wrlen = snprintf( buff+1, sizeof(buff)-1, "%d\n", epoch_name );

    // write metric
    looped_write_to_file( out_metrics_fd, buff, wrlen+1 );
    looped_write_to_file( out_fd, buff, wrlen+1 );
}


/*
 * dump_im_tracing()
 * Dump all in memory tracing information to per-thread files for offline analysis.
 */
void dump_im_tracing() {

    // There are two cases where we want to write to the file. Either we have been told to,
    // because the command server received a command to dump and they wrote the next epoch to
    // shared memory, or because we are shutting down and our model is different than the last time we wrote.
    // Historically, we checked if we were shutting down by testing if t_dump_epoch == shm_data->dump_epoch;
    // However, dump_im_tracing could be called multiple times by the same thread, and the first time that happens
    // we destroy shared memory so shm_data is NULL.
    assert( !(shm_data == NULL && t_dump_epoch == 0) ); // Confirm we never have a case where we just couldn't connect shm
    if( shm_data != NULL && t_dump_epoch < shm_data->dump_epoch ) {
        write_out_tracing();
        t_dump_epoch = shm_data->dump_epoch;
    } else if( shm_data != NULL && t_dump_epoch == shm_data->dump_epoch ) {
        // We are shutting down.
        write_out_tracing();
        // Disconnect shm
        if( is_owner_of_shm ) {
            int rc = shmctl( shmid, IPC_RMID, NULL );
            (void) rc; //suppress unused warning
            assert( rc == 0 );
            rc = sem_unlink( LOCK_SEM_NAME );
            assert( rc == 0 );
            rc = sem_unlink( READY_SEM_NAME );
            assert( rc == 0 );
        }

        shmdt( shm_data );
        sem_close( global_race_sem );
        global_race_sem = NULL;
        sem_close( global_ready_sem );
        global_ready_sem = NULL;
        shm_data = NULL;
    }
}

void stop_libc_monitoring() {
    should_monitor_libc_calls = false;
}

void start_libc_monitoring() {
    should_monitor_libc_calls = true;
}

bool *get_should_monitor_libc_calls() {
    return &should_monitor_libc_calls;
}

ssize_t untracked_write( int fd, const void *buff, size_t sz ) {
    stop_libc_monitoring();
    int wrlen = blessed_write( fd, buff, sz );
    start_libc_monitoring();
    return wrlen;
}
