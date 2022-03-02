from enum import Enum
import signal
import logging
import argparse
import struct
import socket
import sys
import time
import os
import dendrite_analysis
import pyinotify # type: ignore
import numpy as np # type: ignore
import threading
import scipy.stats # type: ignore
import psycopg2 # type: ignore
from model_predictor import ModelArrivalTimePredictor, StratifiedArrivalTimeClassifier
from model_comparison_info import ModelComparisonInfo, ModelHandle
from ConjunctiveRuleEvaluator import ConjunctiveRuleEvaluator
from RuleReader import RuleReader, ParseFailedException # type: ignore
from mousetrap_rule_engine import MousetrapRuleEngine
from rule_register_server import make_register_server_class
from typing import List, Dict, Tuple, Set, Any, Iterable, Optional
from http.server import HTTPServer
from copy import deepcopy
import subprocess
import datetime


log_config = {
    'version': 1,
    'formatters': {
        'detailed': {
            'class': 'logging.Formatter',
            'format': '%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'filename': 'dendrite.log',
            'mode': 'w',
            'formatter': 'detailed'
        }
    },
    'loggers': {
        'command_client': {
            'handlers': [ 'console', 'file' ],
            'level': 'INFO'
        },
        'mousetrap_rule_engine': {
            'handlers': [ 'console', 'file' ],
            'level': 'INFO'
        },
        'dendrite_analysis': {
            'handlers': [ 'console', 'file' ],
            'level': 'INFO'
        },
        'model_predictor': {
            'handlers': [ 'console', 'file' ],
            'level': 'DEBUG'
        }
    }
}
logging.config.dictConfig( log_config )

log = logging.getLogger( 'command_client' )

# Apparently this doesn't get registered by default if we launch this program
# in a backgrounded shell. Without this handler, we don't throw keyboard interrupt exceptions.
signal.signal(signal.SIGINT, signal.default_int_handler)

class ThresholdConfiguration:
    def __init__(
        self,
        probability_threshold: float,
        frequency_threshold: float,
        fingerprint_threshold: float
    ):
        self.probability_threshold = probability_threshold
        self.frequency_threshold = frequency_threshold
        self.fingerprint_threshold = fingerprint_threshold

class VOMComparisonResults:
    def __init__(
        self,
        is_different: bool,
        scaled_results: ModelComparisonInfo,
        unscaled_results: ModelComparisonInfo 
    ):
        self.is_different = is_different
        self.scaled_results = scaled_results
        self.unscaled_results = unscaled_results

class ModelDumpEventHandler( pyinotify.ProcessEvent ): # type: ignore
    """An Inotify handler that monitors /tmp for updated model files and adds them to
       buffers of model data. Also stores combined models for each epoch."""

    def __init__(
        self,
        epoch_interval: int,
        learning_batch_size: int,
        baseline_vom: dendrite_analysis.VariableOrderMarkovGraph,
        baseline_metrics: dendrite_analysis.DstatFileData.DstatMetricData,
        threshold_configuration: ThresholdConfiguration,
        pg_conn: Any,
        exp_id: int
    ) -> None:
        self.epoch_interval = epoch_interval
        self.learning_batch_size = learning_batch_size

        self.baseline_vom = baseline_vom
        self.threshold_configuration = threshold_configuration
        self.pg_conn = pg_conn
        self.exp_id = exp_id
        self.file_models = {} # type: Dict[str,Dict[int,dendrite_analysis.VariableOrderMarkovGraph]]
        self.models_by_epoch = {} # type: Dict[int, List[Tuple[str,dendrite_analysis.VariableOrderMarkovGraph]]]
        self.file_data = {} # type: Dict[str, dendrite_analysis.ModelFileData]
        self.known_model_map = {} # type: Dict[str, str]
        self.models_that_show_up_late = set([]) # type: Set[str]

        self.model_predictors = {} # type: Dict[str, ModelArrivalTimePredictor]
        self.last_epoch_checked = {} # type: Dict[str, int]
        self.unused_model_data_buffers = {} # type: Dict[ str, List[int]]

        self.combined_models_for_epoch = {} # type: Dict[int, dendrite_analysis.VariableOrderMarkovGraph]
        self.all_models_for_epoch = {} # type: Dict[int, List[ModelHandle]]
        self.new_models_for_epoch = {} # type: Dict[int, List[ModelHandle]]

        self.reg_models = {} # type: Dict[str, dendrite_analysis.VariableOrderMarkovGraph]
        self.agg_metrics_for_epoch = {} # type: Dict[int, dendrite_analysis.DstatFileData.DstatMetricData]

        self.dstat_data = None # type: Optional[dendrite_analysis.DstatFileData]
        self.baseline_agg_metrics_data = None # type: Optional[dendrite_analysis.DstatFileData.DstatMetricData]
        self.last_processed_epoch = -1

    def get_dstat_data_for_epoch( self, epoch: int ) -> dendrite_analysis.DstatFileData.DstatMetricData:
        assert epoch in self.agg_metrics_for_epoch
        return self.agg_metrics_for_epoch[ epoch ]

    def get_baseline_dstat_data( self ) -> dendrite_analysis.DstatFileData.DstatMetricData:
        if self.baseline_agg_metrics_data:
            return self.baseline_agg_metrics_data

        # Obtain these from the file

        cur = self.pg_conn.cursor()
        cur.execute( "SELECT cpu_usr, cpu_sys, cpu_idl, cpu_wai, cpu_stl, mem_used, mem_free, mem_buff, mem_cache, net_recv, net_send, disk_read, disk_write, time FROM experiment_epoch_agg_metrics WHERE exp_id = 0 AND epoch_id = 0")
        rows = cur.fetchall()
        assert len(rows) == 1
        row = rows[0]
        csv_fields = [ float(f) for f in row[:13] ]
        t = str(row[13])
        self.baseline_agg_metrics_data = dendrite_analysis.DstatFileData.DstatMetricData( csv_fields, t )
        cur.close()
        assert self.baseline_agg_metrics_data is not None
        return self.baseline_agg_metrics_data

    def set_dstat_data_for_epoch( self, epoch: int, dstat_data: dendrite_analysis.DstatFileData.DstatMetricData ) -> None:
        assert epoch not in self.agg_metrics_for_epoch
        self.agg_metrics_for_epoch[ epoch ] = dstat_data

    def register_model( self, model_name: str, model: dendrite_analysis.VariableOrderMarkovGraph ) -> None:
        self.reg_models[ model_name ] = model

    def add_new_model_fname( self, fname: str ) -> None:
        if fname.endswith( ".im.out" ):
            assert fname not in self.file_models
            self.file_models[ fname ] = {}
        
    def process_IN_CREATE( self, event: Any ) -> None:
        """Process an inotify "file created" event."""
        fname = event.pathname # type: str
        self.add_new_model_fname( fname )

    def do_end_of_epoch_processing( self ) -> None:
        file_names = list(self.file_models.keys())
        print( "End of epoch processing, checking files: {}".format( file_names ), flush=True )
        for fname in file_names:

            if not fname in self.file_data:
                fd  = os.open( fname, os.O_RDONLY )
                path_components = fname.split( "/" )
                base_dir = "/".join( path_components[:-1] )
                end_fname = path_components[-1]
                base_fname = ".".join( end_fname.split(".")[:-2] )
                base_fname = base_fname + ".im.metrics.out"
                constructed_fname = base_dir + "/metrics/" + base_fname
                metrics_fd = os.open( constructed_fname, os.O_RDONLY )
                self.file_data[ fname ] = dendrite_analysis.ModelFileData( fd, metrics_fd )
                self.model_predictors[ fname ] = ModelArrivalTimePredictor( self.epoch_interval )
                self.unused_model_data_buffers[ fname ] = []

            data_buff = b''
            self.file_data[ fname ].process_new_file_data()
            while self.file_data[ fname ].is_ready():
                vom = self.file_data[fname].process_ready_data()
                self.file_models[ fname ][ vom.epoch ] = vom

                if not vom.epoch in self.models_by_epoch:
                    self.models_by_epoch[ vom.epoch ] = []

                self.models_by_epoch[ vom.epoch ].append( ( fname, vom )  )

                unused_data_for_this_model = self.unused_model_data_buffers[ fname ]
                if vom.epoch != 0:
                    unused_data_for_this_model.append( vom.epoch )

                log.info( "Current data buffer for model %s is %s", fname, unused_data_for_this_model )
                if len(unused_data_for_this_model) >= self.learning_batch_size:
                    log.info( "Going to train model for %s", fname )
                    self.model_predictors[fname].add_more_occurrence_data( unused_data_for_this_model )
                    self.unused_model_data_buffers[ fname ] = []

                #if self.model_predictors[fname].classifier is not None:
                #    will_show_up, conf = self.model_predictors[fname].should_show_up( vom.epoch )

                # If we are updating a model that we've already combined...
                if vom.epoch <= self.last_processed_epoch:

                    predictor = self.model_predictors[ fname ]
                    if predictor.classifier is not None:
                        # Identify all historical epochs
                        epochs_in_which_this_model_was_present = self.file_models[ fname ].keys()
                        last_epoch_present = max( [ e for e in epochs_in_which_this_model_was_present if e < vom.epoch ] + [0] )
                        for epoch_to_check in range(vom.epoch,last_epoch_present,-1):
                            should_show, conf = predictor.should_show_up( epoch_to_check )
                            did_show = (vom.epoch == epoch_to_check)
                            log.warning( "Predicted that %s will show in epoch %d: %s, Truth: %s, confidence: %f", fname, epoch_to_check, should_show, did_show, conf )

                    # If there are no models for this epoch, then no combining necessary
                    if not vom.epoch in self.combined_models_for_epoch:
                        self.combined_models_for_epoch[ vom.epoch ] = vom
                        continue 

                    self.models_that_show_up_late.add( fname )

                    # Add this model to the combined model list so we realize that this behaviour happened.
                    # That would normally be handled by the vom comparison code, but they aren't because this is a past epoch.

                    reg_name = determine_vom_type_if_unknown( self, fname, vom, self.threshold_configuration.fingerprint_threshold )
                    model_name = reg_name if reg_name else fname
                    tid = get_tid_from_model_fname( fname )
                    cur = pg_create_cursor( self.pg_conn )
                    cur.execute( "INSERT INTO active_models VALUES( %s, %s, %s, %s )", (self.exp_id, vom.epoch, tid, model_name ) )

                    if self.is_new_vom( fname, vom.epoch ):
                        cur.execute( "INSERT INTO new_models VALUES( %s, %s, %s, %s )", (self.exp_id, vom.epoch, tid, fname) )

                    pg_commit( self.pg_conn, cur )

                    # Combine this model with the combined model from that epoch
                    combined_vom = self.combined_models_for_epoch[ vom.epoch ]
                    combined_vom.merge_in( vom )
                    self.combined_models_for_epoch[ vom.epoch ] = combined_vom # Probably unncessary
                    log.warning( "Updated past epoch by %d adding new info for: %s", vom.epoch, fname )

                    # FIXME: ARGUABLY, we should be re-evaluating the rules here

    def get_merged_vom( self, epoch: int ) -> dendrite_analysis.VariableOrderMarkovGraph:
        """Get the combined model for the given epoch."""
        return self.combined_models_for_epoch[ epoch ]

    def set_merged_vom( self, epoch: int, vom: dendrite_analysis.VariableOrderMarkovGraph ) -> None:
        """Set the combined model for the given epoch."""
        self.combined_models_for_epoch[ epoch ] = vom

    def is_new_vom( self, vom_fname: str, epoch: int )  -> bool:
        cur = pg_create_cursor( self.pg_conn )
        cur.execute( "SELECT 1 FROM new_models WHERE model_file = %s and exp_id = %s", (vom_fname, self.exp_id) )
        rows = cur.fetchall()
        pg_commit( self.pg_conn, cur )
        assert len(rows) < 2
        return len(rows) == 0

    def find_model_start( self, vom_fname: str ) -> Optional[int]:
        cur = pg_create_cursor( self.pg_conn )
        cur.execute( "SELECT epoch_id FROM new_models WHERE model_file = %s and exp_id = %s", (vom_fname, self.exp_id) )
        rows = cur.fetchall()
        pg_commit( self.pg_conn, cur )
        assert len(rows) < 2
        return int(rows[0][0]) if len(rows) == 1 else None


class Command(Enum):
    """Commands that this client can send to the waiting dendrite
    server."""
    DUMP_AND_START_NEW_EPOCH="D"
    RUN_INTERNAL_ENGINE_CMD="I" # engine function

def connect_or_die( host: str, port: int ) -> socket.socket:
    """Connect to the dendrite server running on host:port or die
       with exit status 1."""
    try:
        sock = socket.create_connection( (host, port) )
        return sock
    except ConnectionRefusedError:
        log.error( "Could not connect to %s:%d, terminating...", host, port )
        sys.exit(1)

def send_command( sock: socket.socket, cmd: Command ) -> bool:
    """Send cmd on socket sock. sock should be connected and attached to
    dendrite server process. Returns a bool indicating if the connection is alive."""

    if cmd == Command.DUMP_AND_START_NEW_EPOCH:
        cmd_bytes = cmd.value.encode( 'ascii' )
        log.info( "Sending bytes to socket." )
        sock.send( cmd_bytes )
        log.info( "Sent bytes to socket." )
        data = sock.recv(1) # wait for ACK
        # ACK or hung up?
        return len(data) == 1
    elif cmd == Command.RUN_INTERNAL_ENGINE_CMD:
        cmd_bytes = cmd.value.encode( 'ascii' )
        func_name = "func_to_call"
        func_name_bytes = func_name.encode( 'ascii' )
        func_name_len = len( func_name_bytes )
        func_len_bytes = struct.pack( "<I", func_name_len ) # 4 byte unsigned int
        sock.send( cmd_bytes )
        sock.send( func_len_bytes )
        sock.send( func_name_bytes )
        data = sock.recv(1)
        # ACK?
        return len(data) == 1

    raise ValueError( "Unknown Command Type" )

def create_watchdog(
    epoch_interval: int,
    learning_batch_size: int,
    baseline_vom: dendrite_analysis.VariableOrderMarkovGraph,
    baseline_metrics: dendrite_analysis.DstatFileData.DstatMetricData,
    threshold_configuration: ThresholdConfiguration,
    pg_conn: Any,
    exp_id: int,
    tag_map: Dict[str,dendrite_analysis.FileLocation]
) -> Tuple[ Any, ModelDumpEventHandler]:
    """Set up a background Inotify process that monitors /tmp (not recursively) for
    inotify "IN_MODIFY" events."""

    event_handler = ModelDumpEventHandler( epoch_interval, learning_batch_size, baseline_vom, baseline_metrics, threshold_configuration, pg_conn, exp_id  )
    wm = pyinotify.WatchManager()
    mask = pyinotify.IN_CREATE
    notifier = pyinotify.ThreadedNotifier( wm, event_handler )
    data_path = "/hdd1/dendrite_data"
    full_path_dir_contents = [ os.path.join( data_path, entry ) for entry in os.listdir( data_path ) ]
    existing_file_names = [ entry for entry in full_path_dir_contents if os.path.isfile( entry ) ]
    for fname in existing_file_names:
        event_handler.add_new_model_fname( fname )
    # FIXME: There is a race condition here where a file gets created just before this thread starts, but doesn't happen in our
    # scenarios
    notifier.start()
    wdd = wm.add_watch( data_path, mask, rec=False )
    print( "Added create watch to /hdd1/dendrite_data for in_create.", flush=True )
    return notifier, event_handler


def get_model_handle( handler: ModelDumpEventHandler, model_fname: str ) -> ModelHandle:
    if model_fname in handler.known_model_map:
        return ModelHandle( model_fname, handler.known_model_map[ model_fname ] )
    return ModelHandle( model_fname, model_fname )

def remap_model_list_to_registered_model_list( handler: ModelDumpEventHandler,
        model_name_list: List[str] ) -> List[ModelHandle]:
    """Look at the fnames and check if the previous model we knew about for that fname
    corresponded to a registered model type.
    Use this registered model type as a nickname for the model"""

    model_handles = [] # type: List[ModelHandle]
    for model_fname in model_name_list:
        model_handles.append( get_model_handle( handler, model_fname ) )
    assert len( model_handles ) == len( model_name_list )
    return model_handles

def remap_ci_violation_data_fnames_to_reg_names(
    handler: ModelDumpEventHandler,
    model_ci_viols: List[ModelComparisonInfo.CIViolationData]
)-> None:
    model_fnames = [ model_ci_viol.model_handle.get_fname() for model_ci_viol in model_ci_viols ]
    model_handles = remap_model_list_to_registered_model_list( handler, model_fnames )
    assert len(model_fnames) == len(model_handles)
    for i in range(len(model_ci_viols)):
        model_ci_viols[i].set_model_nickname( model_handles[i].get_nickname() )

def level_missing_mapper_dendrite_original(
    loc: dendrite_analysis.FileLocation,
    level: int,
    count: int,
    events1: Dict[dendrite_analysis.FileLocation, dendrite_analysis.EventRecord],
    events2: Dict[dendrite_analysis.FileLocation, dendrite_analysis.EventRecord] ) -> float:

    txns1 = 0
    txns2 = 0
    txn_loc = dendrite_analysis.FileLocation("xact.c", 2084)
    if txn_loc in events1:
        txns1 = events1[txn_loc].count
    if txn_loc in events2:
        txns2 = events2[txn_loc].count
    min_txns = min(txns1, txns2)
    max_txns = max(txns1, txns2)

    #if max_txns / min_txns <= 1.25:
        # We don't care unless performance tanks thank you.
    #    return 1.0
 
    #if loc.fname == 'postinit.c':
    #    return 10.0
    if loc.fname == 'md.c' or loc.fname == 'bufmgr.c':
        return count/100.
    if (not loc.fname == 'postinit.c' and not loc.fname == 'postgres.c' and level < 14) or loc.fname == 'predicate.c' or loc.fname == 'nodeLockRows.c' or loc.fname == 'nodeModifyTable.c':
        return 1.0
    return 5.0

def level_missing_mapper_dendrite_ng_fp(
    loc: dendrite_analysis.FileLocation,
    level: int,
    count: int,
    events1: Dict[dendrite_analysis.FileLocation, dendrite_analysis.EventRecord],
    events2: Dict[dendrite_analysis.FileLocation, dendrite_analysis.EventRecord] ) -> float:
    """Log level remapper for dendrite ng. For now, assuming that any missing event is important because there
    aren't that many event types that we capture. Always return 5.0"""

    txns1 = 0
    txns2 = 0
    txn_loc = dendrite_analysis.FileLocation("xact.c", 2084)
    if txn_loc in events1:
        txns1 = events1[txn_loc].count
    if txn_loc in events2:
        txns2 = events2[txn_loc].count
    min_txns = min(txns1, txns2)
    max_txns = max(txns1, txns2)

    # Small buffer/md diffs don't matter that much. Compress how much these contribute
    if loc.fname == 'md.c' or loc.fname == 'bufmgr.c':
        if loc in events1 and events1[loc].count < 1000:
            return 2.0
        if loc in events2 and events2[loc].count < 1000:
            return 2.0
        # Fall through


    if loc.fname == 'smgrread':
        loc = dendrite_analysis.FileLocation("smgrread", 0)
        if loc in events1 and events1[loc].count < 500:
            return 2.0
        if loc in events2 and events2[loc].count < 500:
            return 2.0
        # Fall through

    if loc.fname == 'CreateCheckPoint':
        # Not a checkpointer unless you have this
        return 100.0

    return 5.0


def level_missing_mapper_dendrite_ng(
    loc: dendrite_analysis.FileLocation,
    level: int,
    count: int,
    events1: Dict[dendrite_analysis.FileLocation, dendrite_analysis.EventRecord],
    events2: Dict[dendrite_analysis.FileLocation, dendrite_analysis.EventRecord] ) -> float:
    """Log level remapper for dendrite ng. For now, assuming that any missing event is important because there
    aren't that many event types that we capture. Always return 5.0"""

    txns1 = 0
    txns2 = 0
    txn_loc = dendrite_analysis.FileLocation("xact.c", 2084)
    if txn_loc in events1:
        txns1 = events1[txn_loc].count
    if txn_loc in events2:
        txns2 = events2[txn_loc].count
    min_txns = min(txns1, txns2)
    max_txns = max(txns1, txns2)

    # Total count
    tot1 = 0
    md_count1 = 0
    pgstat_count1 = 0
    for k in events1.keys():
        tot1 += events1[k].count
        if k.fname == 'md.c' or k.fname == 'bufmgr.c':
            md_count1 += events1[k].count
        if k.fname == 'pgstat.c':
            pgstat_count1 += events1[k].count

    tot2 = 0
    md_count2 = 0
    pgstat_count2 = 0
    for k in events2.keys():
        tot2 += events2[k].count
        if k.fname == 'md.c' or k.fname == 'bufmgr.c':
            md_count2 += events2[k].count
        if k.fname == 'pgstat.c':
            pgstat_count2 += events2[k].count

    if pgstat_count1/tot1 > 0.8 or pgstat_count2/tot2 > 0.8:
        # This is a model check, don't count the stat collector
        return 5.0


    if loc.fname == "pgstat.c" or loc.fname == 'xlog.c':
        return 1.0


    # Small buffer/md diffs don't matter that much. Compress how much these contribute
    if loc.fname == 'md.c' or loc.fname == 'bufmgr.c':
        if loc in events1 and events1[loc].count < 1000:
            return 2.0
        if loc in events2 and events2[loc].count < 1000:
            return 2.0
        # Fall through


    if loc.fname == 'smgrread':
        loc = dendrite_analysis.FileLocation("smgrread", 0)
        if loc in events1 and events1[loc].count < 500:
            return 2.0
        if loc in events2 and events2[loc].count < 500:
            return 2.0
        # Fall through

    if loc.fname == 'CreateCheckPoint':
        # Not a checkpointer unless you have this
        return 100.0

    return 5.0



def log_level_score_rescaler_dendrite_original(
    loc: dendrite_analysis.FileLocation,
    level: float,
    score: float,
    events1: Dict[dendrite_analysis.FileLocation, dendrite_analysis.EventRecord],
    events2: Dict[dendrite_analysis.FileLocation, dendrite_analysis.EventRecord] 
 ) -> float:

    txns1 = 0
    txns2 = 0
    txn_loc = dendrite_analysis.FileLocation("xact.c", 2084)
    if txn_loc in events1:
        txns1 = events1[txn_loc].count
    if txn_loc in events2:
        txns2 = events2[txn_loc].count
    min_txns = min(txns1, txns2)
    max_txns = max(txns1, txns2)


    #if max_txns / min_txns <= 1.25:
        # We don't care unless performance tanks thank you.
    #    return 1.0
    
    #if loc.fname == 'postinit.c':
    #    return 10.0
    #if loc.fname == 'lock.c':
    #    return score * 2
    # Maybe
    #if loc.fname == 'md.c':
    #    return score / 2 
    if (not loc.fname == 'postinit.c' and not loc.fname == 'postgres.c' and not loc.fname == 'bufmgr.c' and level < 14) or loc.fname == 'predicate.c' or loc.fname == 'nodeLockRows.c' or loc.fname == 'nodeModifyTable.c':
        return 1.0
    return score

def log_level_score_rescaler_dendrite_ng(
    loc: dendrite_analysis.FileLocation,
    level: float,
    score: float,
    events1: Dict[dendrite_analysis.FileLocation, dendrite_analysis.EventRecord],
    events2: Dict[dendrite_analysis.FileLocation, dendrite_analysis.EventRecord] 
 ) -> float:
    """Log level rescaler for dendrite ng. For now, assume everything has the same weight. This is
    because in dendrite NG, events are hand-selected so they are presumably all important. If there
    are some spurious one-off events causing issues, we can prune those later."""

    txns1 = 0
    txns2 = 0
    txn_loc = dendrite_analysis.FileLocation("CommitTransaction", 0)
    if txn_loc in events1:
        txns1 = events1[txn_loc].count
    if txn_loc in events2:
        txns2 = events2[txn_loc].count
    txn_loc = dendrite_analysis.FileLocation("xact.c",2206)
    if txn_loc in events1:
        txns1 = events1[txn_loc].count
    if txn_loc in events2:
        txns2 = events2[txn_loc].count
    min_txns = min(txns1, txns2)
    max_txns = max(txns1, txns2)


    # Total count
    tot1 = 0
    md_count1 = 0
    pgstat_count1 = 0
    for k in events1.keys():
        tot1 += events1[k].count
        if k.fname == 'md.c' or k.fname == 'bufmgr.c':
            md_count1 += events1[k].count
        if k.fname == 'pgstat.c':
            pgstat_count1 += events1[k].count

    tot2 = 0
    md_count2 = 0
    pgstat_count2 = 0
    for k in events2.keys():
        tot2 += events2[k].count
        if k.fname == 'md.c' or k.fname == 'bufmgr.c':
            md_count2 += events2[k].count
        if k.fname == 'pgstat.c':
            pgstat_count2 += events2[k].count

    if pgstat_count1/tot1 > 0.8 or pgstat_count2/tot2 > 0.8:
        # This is a model check, don't count the stat collector
        return 5.0


    if loc.fname != "md.c" and loc.fname != "bufmgr.c" and float(md_count1)/tot1 > 0.8 and float(md_count2)/tot2 > 0.8:
        tot1 -= md_count1 
        tot2 -= md_count2
        frac1 = float(events1[loc].count)/tot1
        frac2 = float(events2[loc].count)/tot2
        return frac1/frac2 if frac1 >= frac2 else frac2/frac1

    # Add compression for dendrite log
    if loc.fname == "smgrread" or loc.fname == "smgrwrite" or loc.fname == "md.c" or loc.fname == "bufmgr.c":
        # These events can be triggerd by background flushing, checkpointing, autovac.
        # They can result iin a huge burst of disk activity, but we care only if we hamper
        # txn performance. Compress the range unless txns differ by more than 20%
        if min_txns > 0 and float(max_txns)/min_txns <= 1.2:
            return min(2.0,score)
    if loc.fname == "pgstat.c":
        return 1.0

    return score

def check_model_arrival_times(
    handler: ModelDumpEventHandler,
    epoch: int,
    prev_epoch: int,
    new_voms: List[ModelHandle]
) -> Tuple[bool,Optional[ModelComparisonInfo]]:
    """Check if models showed up within their confidence intervals for this epoch."""
    return (False, None)

def low_level_vom_diff(
    vom1: dendrite_analysis.VariableOrderMarkovGraph,
    vom2: dendrite_analysis.VariableOrderMarkovGraph,
    apply_scaling: bool
) -> Tuple[List[dendrite_analysis.EventDifferenceRecord],List[dendrite_analysis.VOTransitionDiffRecord]]:
    """Do a VOM comparison, rescaling missing results and based on log levels. Else, use 5.0"""

    if apply_scaling:
        return vom1.diff( vom2, level_missing_mapper_dendrite_ng, log_level_score_rescaler_dendrite_ng )
    return vom1.diff( vom2, lambda *x: 5.0, log_level_score_rescaler_dendrite_ng )

def single_vom_comparison(
    vom1: dendrite_analysis.VariableOrderMarkovGraph,
    vom2: dendrite_analysis.VariableOrderMarkovGraph,
    threshold_configuration: ThresholdConfiguration,
    all_voms: List[ModelHandle],
    new_voms: List[ModelHandle],
    apply_scaling: bool = False
) -> Tuple[bool, ModelComparisonInfo]:

    event_difference_records, vot_diff_records = low_level_vom_diff( vom1, vom2, apply_scaling )

    mean_prob_diff, mean_count_diff = dendrite_analysis.compute_event_difference_stats( event_difference_records )

    log.info( "Determined that mean_prob_diff is: %f", mean_prob_diff )
    #log.info( "Determined that mean_count_diff is: %f", mean_count_diff )
    is_different = mean_prob_diff > threshold_configuration.probability_threshold #or mean_count_diff > threshold_configuration.frequency_threshold

    is_different_as_rt = ModelComparisonInfo.ComparisonResultType.MODELS_NOT_SIMILAR if is_different else ModelComparisonInfo.ComparisonResultType.MODELS_SIMILAR
    print( "VOM comparison, is scaled: {}, is different: {}".format( apply_scaling, is_different ) )

    return is_different, ModelComparisonInfo(
        is_different_as_rt,
        all_models=all_voms,
        new_models=new_voms,
        event_difference_records=event_difference_records,
        transition_difference_records=vot_diff_records
    )

def compare_voms(
    vom1: dendrite_analysis.VariableOrderMarkovGraph,
    vom2: dendrite_analysis.VariableOrderMarkovGraph,
    threshold_configuration: ThresholdConfiguration,
    all_voms: List[ModelHandle],
    new_voms: List[ModelHandle],
    output: bool = True
) -> VOMComparisonResults:

    is_scaled_different, scaled_mci = single_vom_comparison(
        vom1,
        vom2,
        threshold_configuration,
        all_voms,
        new_voms,
        apply_scaling=True
    )

    is_unscaled_different, unscaled_mci = single_vom_comparison(
        vom1,
        vom2,
        threshold_configuration,
        all_voms,
        new_voms,
        apply_scaling=False
    )

    return VOMComparisonResults( is_scaled_different, scaled_mci, unscaled_mci )

def determine_vom_type_if_unknown( handler: ModelDumpEventHandler, fname: str, vom: dendrite_analysis.VariableOrderMarkovGraph, threshold: float ) -> Optional[str]:
    if not fname in handler.known_model_map:
        for reg_vom_name, reg_vom in handler.reg_models.items():
            same_model = reg_vom.is_similar_to( vom, threshold, level_missing_mapper_dendrite_ng, log_level_score_rescaler_dendrite_ng )
            if same_model:
                handler.known_model_map[ fname ] = reg_vom_name
                log.info( "Determined that %s is of type: %s", fname, reg_vom_name )
                return reg_vom_name
        if not fname in handler.known_model_map:
            #handler.known_model_map[ fname ] = "Unknown"
            log.warning( "Unknown model type for %s", fname )
    return None

def process_models_for_epoch(
    handler: ModelDumpEventHandler,
    models_for_epoch: List[Tuple[str,dendrite_analysis.VariableOrderMarkovGraph]],
    epoch: int,
    threshold_configuration: ThresholdConfiguration
) -> Tuple[List[str], List[int]]:

    new_vom_fnames = []

    # If we fingerprint models, we should figure out when those models were new and re-evaluate the rules
    # in that epoch if behaviour differed
    fingerprinted_new_model = False
    newly_fingerprinted_model_starts = []

    # Copy so we don't clobber the model
    fname, model = models_for_epoch[0]
    vom = deepcopy( model )
    new_vom = handler.is_new_vom( fname, epoch )
    if new_vom:
        new_vom_fnames.append( fname )

    # Check what this vom corresponds to
    name = determine_vom_type_if_unknown(
        handler,
        fname,
        vom,
        threshold_configuration.fingerprint_threshold
    )
    if name:
        fingerprinted_new_model = True
        if new_vom:
            newly_fingerprinted_model_starts.append( epoch )
        else:
            model_start = handler.find_model_start( fname )
            assert model_start is not None
            # Can be none because we havent' seen this before.
            newly_fingerprinted_model_starts.append( model_start )

    # Merge everything in.
    for next_name_and_vom in models_for_epoch[1:]:
        next_fname, next_vom = next_name_and_vom
        vom.merge_in( next_vom )
        new_vom = handler.is_new_vom( next_fname, epoch )
        if new_vom:
            new_vom_fnames.append( next_fname )
        reg_name = determine_vom_type_if_unknown(
            handler,
            next_fname,
            next_vom,
            threshold_configuration.fingerprint_threshold
        )
        if reg_name:
            fingerprinted_new_model = True
            if new_vom:
                newly_fingerprinted_model_starts.append( epoch )
            else:
                model_start = handler.find_model_start( next_fname )
                assert model_start is not None
                newly_fingerprinted_model_starts.append( model_start )

    # Set done processing, update epoch
    handler.set_merged_vom( epoch, vom )

    log.warning( "Newly Fingerprinted model starts: %s", newly_fingerprinted_model_starts )

    return new_vom_fnames, newly_fingerprinted_model_starts

def reevaluate_past_epoch_rules_if_necessary(
    handler: ModelDumpEventHandler,
    newly_fingerprinted_model_starts: List[int],
    epoch: int,
    threshold_configuration: ThresholdConfiguration,
    mte: MousetrapRuleEngine,
    cur: Any
) -> None:

    model_start_set = set(newly_fingerprinted_model_starts)
    if epoch in model_start_set:
        model_start_set.remove( epoch )
    log.warning( "After removing epoch %d, have %s", epoch, model_start_set )
    for reeval_epoch in model_start_set:
        if reeval_epoch == 0:
            continue
        log.warning( "Re-evaluating rules from epoch %d because we have new fingerprints for models that started then!", reeval_epoch )
        old_model = handler.combined_models_for_epoch[ reeval_epoch ]
        all_models_fnames = [ mh.get_fname() for mh in handler.all_models_for_epoch[ reeval_epoch ] ]
        all_models = remap_model_list_to_registered_model_list( handler, all_models_fnames )
        new_models_fnames = [ mh.get_fname() for mh in handler.new_models_for_epoch[ reeval_epoch ] ]
        new_models = remap_model_list_to_registered_model_list( handler, new_models_fnames )
        prev_agg_metrics = handler.agg_metrics_for_epoch[ reeval_epoch-1 ]
        cur_agg_metrics = handler.agg_metrics_for_epoch[ reeval_epoch ]
        
        log.info( "Doing VOM comparison, baseline against epoch %s, to check for re-eval.",  reeval_epoch )
        vom_comp_results = compare_voms(
            handler.baseline_vom,
            old_model,
            threshold_configuration,
            all_models,
            new_models,
            output=True
        )

        baseline_similar, baseline_diff_info = not vom_comp_results.is_different, vom_comp_results.scaled_results
        log.info( "Re-evaluation got similar: %s, type: %s", baseline_similar, baseline_diff_info.comparison_result_type )

        if baseline_diff_info.comparison_result_type == ModelComparisonInfo.ComparisonResultType.MODELS_SIMILAR:
            log.info( "Got new, active models: %s %s", new_models, all_models )
            eval_result = mte.evaluate_all_rules( baseline_diff_info, prev_agg_metrics, cur_agg_metrics )
            if eval_result.rule_was_fired():
                # FIXME: should be upsert/join
                cur.execute( "INSERT INTO rules_fired VALUES ( %s, %s, %s )", (handler.exp_id, reeval_epoch, '\n'.join( eval_result.get_fired_rules() ) ) )

        log.info( "REVISED Outcome for epoch %d: %s", reeval_epoch, baseline_diff_info.comparison_result_type )

def combine_current_epoch_models(
        handler: ModelDumpEventHandler,
        epoch: int,
        threshold_configuration: ThresholdConfiguration,
        mte: MousetrapRuleEngine,
        cur: Any
) -> Tuple[List[str],List[int]]:
    """Get all the models from the current epoch that have been dumped.
    Combine them together, store them."""
    log.info( "Epoch %d: Enough time has passed, going to check for all available models.", epoch )

    models_for_cur_epoch = [] # type: List[Tuple[str,dendrite_analysis.VariableOrderMarkovGraph]]

    # Get all the models we have for this epoch and the previous one
    if epoch in handler.models_by_epoch:
        models_for_cur_epoch = handler.models_by_epoch[epoch]

    # Flag to check if this is the first time we've seen a particular VOM

    all_model_names = list([ m[0] for m in models_for_cur_epoch ])
    all_models = remap_model_list_to_registered_model_list( handler, all_model_names )

    vom = None

    if models_for_cur_epoch:
        new_vom_fnames, newly_fingerprinted_model_starts = process_models_for_epoch(
            handler,
            models_for_cur_epoch,
            epoch,
            threshold_configuration

        )
        return new_vom_fnames, newly_fingerprinted_model_starts
    else:
        log.warning( "There are no models for the current epoch!" )
        raise Exception( "WTF" )


def check_model_arrivals(
    handler: ModelDumpEventHandler,
    epoch: int
) -> Tuple[bool,Optional[ModelComparisonInfo]]:

    assert epoch in handler.models_by_epoch
    all_known_fnames = list(handler.file_models.keys())
    models_in_this_epoch = [ fname_model[0] for fname_model in handler.models_by_epoch[ epoch ] ]
    log.info( "We obtained these models in this epoch: %s", models_in_this_epoch )

    early_models = []
    late_models = []

    for fname in all_known_fnames:
        if not fname in handler.model_predictors:
            log.warning( "No model predictor for %s found!", fname )
            continue
        predictor = handler.model_predictors[ fname ]
        # If this model does not show up in this epoch and it is known to not show up on time,
        # don't read too much into it.
        # Unless it hasn't shown up for a while...
        if fname not in models_in_this_epoch: #and fname in handler.models_that_show_up_late:

            # When was this model last active ?
            epochs_in_which_this_model_was_present = handler.file_models[ fname ].keys()
            last_epoch_present = max( [ e for e in epochs_in_which_this_model_was_present if e < epoch ] + [0] )

            if predictor.classifier is not None and isinstance( predictor.classifier, StratifiedArrivalTimeClassifier ):
                periods = [ l.period_with_offset.period for l in predictor.classifier.learners ]
                log.info( "GOT ALL PERIODS FOR LEANERS %s", periods )
                if epoch - last_epoch_present <= min(periods):
                    log.debug( "Not predicting for %s, known to show up late. Will be handled by model dump loop.", fname )
                    continue
                last_epoch_checked = last_epoch_present
                if fname in handler.last_epoch_checked:
                    last_epoch_checked = handler.last_epoch_checked[ fname ]
                for epoch_to_check in range(epoch,last_epoch_checked,-1):
                    should_show, conf = predictor.should_show_up( epoch_to_check )
                    did_show = (epoch_to_check in handler.file_models[ fname ])
                    log.info( "%d: Prediction for %s is %s; Truth is %s. Confidence: %f", epoch_to_check, fname, should_show, did_show, conf )
                    if should_show != did_show and conf >= 0.8:
                        log.warning( "PRED IS BROKEN!" )
                        handle = get_model_handle( handler, fname )
                        ci_viol_data = ModelComparisonInfo.CIViolationData( handle, 0., 0., 0. )
                        if should_show:
                            late_models.append( ci_viol_data )
                        else:
                            early_models.append( ci_viol_data )
                        break
                handler.last_epoch_checked[ fname ] = epoch
                continue

            elif last_epoch_present == epoch-1:
                # Maybe showed up late just this once?
                continue

        if predictor.classifier is not None:
            should_show, conf = predictor.should_show_up( epoch )
            did_show = fname in models_in_this_epoch
            log.info( "Prediction for %s is %s; Truth is %s. Confidence: %f", fname, should_show, did_show, conf )
            if should_show != did_show and conf >= 0.8:
                log.warning( "PRED IS BROKEN!" )
                handle = get_model_handle( handler, fname )
                ci_viol_data = ModelComparisonInfo.CIViolationData( handle, 0., 0., 0. )
                if should_show:
                    late_models.append( ci_viol_data )
                else:
                    early_models.append( ci_viol_data )
    if early_models or late_models:
        mci = ModelComparisonInfo( ModelComparisonInfo.ComparisonResultType.CI_VIOLATING_MODELS, late_models=late_models, early_models=early_models )
        return (True, mci)
    return (False,None)

def do_vom_comparisons_for_cur_epoch(
    handler: ModelDumpEventHandler,
    epoch: int,
    exp_id: int,
    threshold_configuration: ThresholdConfiguration,
    mte: MousetrapRuleEngine,
    cur: Any,
    new_vom_fnames: List[str],
    newly_fingerprinted_model_starts: List[int]
) -> Tuple[bool, ModelComparisonInfo]:

    assert epoch in handler.combined_models_for_epoch
    handler.last_processed_epoch = epoch
    vom = handler.combined_models_for_epoch[ epoch ]

    # Get all the models we have for this epoch
    models_for_cur_epoch = handler.models_by_epoch[epoch]

    all_model_names = list([ m[0] for m in models_for_cur_epoch ])
    all_models = remap_model_list_to_registered_model_list( handler, all_model_names )

    new_voms = remap_model_list_to_registered_model_list( handler, new_vom_fnames )
    log.debug( "Found new voms: %s", new_voms )

    # If this is the first epoch, we won't have a previous model to compare against
    if epoch == 0:
        log.warning( "Not comparing model because its the first epoch." )
        return (True, ModelComparisonInfo(ModelComparisonInfo.ComparisonResultType.NOT_ENOUGH_DATA,
            all_models=all_models, new_models=new_voms))

    reevaluate_past_epoch_rules_if_necessary(
        handler,
        newly_fingerprinted_model_starts,
        epoch,
        threshold_configuration,
        mte,
        cur
    )

    models_violate_cis, ci_model_comparison_info = check_model_arrivals(
        handler,
        epoch,
    )

    # If this is not the first epoch, compare it to the previous epoch's model
    prev_epoch = epoch-1

    # Get the combined model for the prev epoch, if we have one
    combined_model_for_prev_epoch = handler.get_merged_vom( prev_epoch )

    # models_violate_cis should not directly count as a behavioural difference.
    if vom:
        prev_vom = handler.get_merged_vom( prev_epoch )
        if not prev_vom:
            log.warning( "No previous model to compare behaviour against" )
            if models_violate_cis:
                assert not ci_model_comparison_info is None
                return (True, ci_model_comparison_info)
            return (True,ModelComparisonInfo(ModelComparisonInfo.ComparisonResultType.NOT_ENOUGH_DATA,
                all_models=all_models, new_models=new_voms) )

        log.info( "Comparing model for epoch %s against epoch %s", epoch, prev_epoch )
        vom_comp_results = compare_voms(
            vom,
            prev_vom,
            threshold_configuration,
            all_models,
            new_voms
        )

        models_similar, comp_info = not vom_comp_results.is_different, vom_comp_results.scaled_results

        cur_vom = handler.combined_models_for_epoch[ epoch ]

        pg_record_vom( cur, exp_id, epoch, cur_vom )
        pg_record_prev_epoch_diffs( cur, exp_id, epoch, models_similar, vom_comp_results.unscaled_results, models_violate_cis, ci_model_comparison_info )

        # If we have CI violations, stich those into the results as well.
        if models_similar:
            if models_violate_cis:
                assert not ci_model_comparison_info is None
                return (False, ci_model_comparison_info)
            return (True, comp_info)
        assert comp_info is not None
        if models_violate_cis:
            assert ci_model_comparison_info is not None
            tmp_info = comp_info # type: ModelComparisonInfo
            comp_info.early_models = ci_model_comparison_info.early_models
            comp_info.late_models = ci_model_comparison_info.late_models
            return (False, comp_info)
        return (False, comp_info)

    else:
        log.warning( "Could not obtain model for epoch {}".format( epoch ) )
        if models_violate_cis:
            assert not ci_model_comparison_info is None
            return (False, ci_model_comparison_info)
        return (True,ModelComparisonInfo(ModelComparisonInfo.ComparisonResultType.NOT_ENOUGH_DATA,
            all_models=all_models, new_models=new_voms) )

def create_epoch_in_pg( cur: Any, exp_id: int, epoch: int ) -> None:
    cur.execute( "INSERT INTO experiment_epochs VALUES( %s, %s, %s )", (exp_id, epoch, datetime.datetime.now()) )

def record_agg_metrics_in_pg( cur: Any, exp_id: int, epoch: int, cur_agg_metrics: dendrite_analysis.DstatFileData.DstatMetricData ) -> None:
    cur.execute( "INSERT INTO experiment_epoch_agg_metrics VALUES( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s )",
        (exp_id, epoch) + cur_agg_metrics.get_pg_tup() );

def pg_create_cursor( pg_conn: Any ) -> Any:
    return pg_conn.cursor()

def pg_commit( pg_conn: Any, cur: Any ) -> Any:
    pg_conn.commit()
    cur.close()

def pg_record_epoch_and_agg_metrics( cur: Any, exp_id: int, epoch: int, cur_agg_metrics: dendrite_analysis.DstatFileData.DstatMetricData ) -> None:
    create_epoch_in_pg( cur, exp_id, epoch )
    record_agg_metrics_in_pg( cur, exp_id, epoch, cur_agg_metrics )

def pg_record_vom( cur: Any, exp_id: int, epoch: int, vom: dendrite_analysis.VariableOrderMarkovGraph ) -> None:
    cur.execute( "INSERT INTO serialized_voms VALUES( %s, %s, %s )", (exp_id, epoch, vom.serialize() ) )

def get_tid_from_model_fname( fname: str ) -> int:
    return int( fname.split(".")[1].split("/")[-1] )

def pg_record_prev_epoch_diffs(
    cur: Any,
    exp_id: int,
    epoch: int,
    models_similar: bool,
    difference_info: ModelComparisonInfo,
    models_violate_cis: bool,
    ci_model_comparison_info: ModelComparisonInfo
 ) -> None:
    for new_model in difference_info.new_models:
        tid = get_tid_from_model_fname( new_model.get_fname() )
        cur.execute( "INSERT INTO new_models VALUES( %s, %s, %s, %s )", (exp_id, epoch, tid, new_model.get_fname() ) )

    for model_name in difference_info.all_models:
        tid = get_tid_from_model_fname( model_name.get_fname() )
        cur.execute( "INSERT INTO active_models VALUES( %s, %s, %s, %s )", (exp_id, epoch, tid, model_name.get_nickname() ) )

    if models_violate_cis:
        for early_model_data in ci_model_comparison_info.early_models:
            tid = get_tid_from_model_fname( early_model_data.model_handle.get_fname() )
            cur.execute( "INSERT INTO ci_violations VALUES( %s, %s, %s, %s, %s, %s )", (exp_id, epoch, tid, early_model_data.predictor_mean, early_model_data.predictor_variance, early_model_data.arrival_time ) )

        for late_model_data in ci_model_comparison_info.late_models:
            tid = get_tid_from_model_fname( late_model_data.model_handle.get_fname() )
            cur.execute( "INSERT INTO ci_violations VALUES( %s, %s, %s, %s, %s, %s )", (exp_id, epoch, tid, late_model_data.predictor_mean, late_model_data.predictor_variance, late_model_data.arrival_time ) )

    if difference_info.comparison_result_type != ModelComparisonInfo.ComparisonResultType.NOT_ENOUGH_DATA:
        for p_diff in difference_info.event_difference_records:
            sql_txt = "INSERT INTO vom_prob_diffs VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            sql_args = (exp_id, epoch, exp_id, epoch-1, p_diff.event_fname, p_diff.event_line,
                p_diff.left_count, p_diff.right_count, p_diff.left_prob, p_diff.right_prob )
            cur.execute( sql_txt, sql_args )

        cur.execute( "INSERT INTO vom_comparison_results VALUES ( %s, %s, %s, %s, %s, %s )", (exp_id, epoch, exp_id, epoch-1, not models_similar, len(difference_info.early_models) > 0 or len(difference_info.late_models) > 0 ) )

def pg_record_baseline_diffs( cur: Any, exp_id: int, epoch: int, baseline_similar: bool, baseline_diff_info: ModelComparisonInfo ) -> None:
    baseline_exp_id = 0
    baseline_epoch_id = 0

    for p_diff in baseline_diff_info.event_difference_records:
        sql_txt = "INSERT INTO vom_prob_diffs VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        sql_args = (exp_id, epoch, baseline_exp_id, baseline_exp_id, p_diff.event_fname, p_diff.event_line,
               p_diff.left_count, p_diff.right_count, p_diff.left_prob, p_diff.right_prob )

        cur.execute( sql_txt, sql_args )

    cur.execute( "INSERT INTO vom_comparison_results VALUES ( %s, %s, %s, %s, %s, %s )", (exp_id, epoch, baseline_exp_id, baseline_exp_id, not baseline_similar, False) )

def process_and_obtain_dstat_metrics(
        handler: ModelDumpEventHandler,
        epoch: int,
        dstat_fname: str ) -> Tuple[dendrite_analysis.DstatFileData.DstatMetricData, Optional[dendrite_analysis.DstatFileData.DstatMetricData], dendrite_analysis.DstatFileData.DstatMetricData]:

     # Get aggregate statistics
    if not handler.dstat_data:
        dstat_fd = os.open( dstat_fname, os.O_RDONLY )
        dstat_data = dendrite_analysis.DstatFileData( dstat_fd )
        handler.dstat_data = dstat_data
    assert handler.dstat_data

    new_data = handler.dstat_data.process_new_file_data()
    assert new_data


    cur_agg_metrics = handler.dstat_data.process_ready_data()
    handler.set_dstat_data_for_epoch( epoch, cur_agg_metrics )

    prev_agg_metrics = handler.get_dstat_data_for_epoch( epoch-1 ) if epoch > 0 else None

    baseline_agg_metrics = handler.get_baseline_dstat_data()

    return cur_agg_metrics, prev_agg_metrics, baseline_agg_metrics

def do_vom_comparisons_against_baseline(
    handler: ModelDumpEventHandler,
    epoch: int,
    exp_id: int,
    prev_agg_metrics: Optional[dendrite_analysis.DstatFileData.DstatMetricData],
    cur_agg_metrics: dendrite_analysis.DstatFileData.DstatMetricData,
    cur_vom: dendrite_analysis.VariableOrderMarkovGraph,
    threshold_configuration: ThresholdConfiguration,
    all_models: List[ModelHandle],
    new_models: List[ModelHandle],
    cur: Any
) -> Tuple[bool, ModelComparisonInfo]:

    vom_comp_results =  compare_voms( handler.baseline_vom, cur_vom, threshold_configuration, all_models, new_models, output=True )

    baseline_similar, baseline_diff_info = not vom_comp_results.is_different, vom_comp_results.unscaled_results
    pg_record_baseline_diffs( cur, exp_id, epoch, baseline_similar, baseline_diff_info )
    log.info( "Determined from vom comparison that baseline is similar to epoch: %d -> %s", epoch, baseline_similar )

    return baseline_similar, vom_comp_results.scaled_results

def contrast_models_and_respond(
        handler: ModelDumpEventHandler,
        pg_conn: Any,
        mte: MousetrapRuleEngine,
        exp_id: int,
        epoch: int,
        threshold_configuration: ThresholdConfiguration,
        dool_file: str
) -> None:

    handler.do_end_of_epoch_processing()

    cur_agg_metrics, prev_agg_metrics, baseline_agg_metrics = process_and_obtain_dstat_metrics( handler, epoch, dool_file )

    cur = pg_create_cursor( pg_conn )
    pg_record_epoch_and_agg_metrics( cur, exp_id, epoch, cur_agg_metrics )

    new_vom_fnames, newly_fingerprinted_model_starts = combine_current_epoch_models(
        handler,
        epoch,
        threshold_configuration,
        mte,
        cur
    )

    # Caveat:
    # It is possible for some processes to be active in a given epoch but then not be part of the models
    # outputted. This scenario happens when a process was active and doing stuff, but then went to sleep and did not
    # wake up until a later epoch. There are no bounds on how late we may then learn about the activity in a prior epoch:
    # we may sleep in epoch 9 and only output the behaviour for what we did in epoch 9 during epoch 100.
    #
    # This complication poses two key challenges:
    # 1) During VOM comparison, we may have a processes' model included, but due to timing conditions, that model has
    #    not yet arrived in the current epoch. If so, we may presume there is a behaviour difference where there is none
    # 2) If we are trying to predict model arrival times, we may presume that a model has not been active in a given epoch,
    #    when really it was. The process has merely gone to sleep and so we do not have its information.
    #
    # How should we deal with these issues?
    # Option 1: We ignore any information that comes in late, never merge it into the models or consider it as part of behaviour
    #           differences. This might appear to reduce race conditions, but we just move the race condition to if the model dumps
    #           before we observe it
    # Option 2: We integrate the behaviour information, and accept that we may occasionally be affected by race conditions and log it.
    #           If we observe it being an issue, then we do something about it
    # Option 3: We recognize that some models may arrive late and never consider those models are part of behaviour??
    #
    # We go with option 2.
    #
    # Regarding preds, we could just do the preds once we obtain a new model instance and then check if it should have appeared at any of
    # the prior times (again only for models that show up late)...
    log.info( "Doing epoch vs prev epoch comparison, at epoch %d", epoch )
    models_similar, difference_info = do_vom_comparisons_for_cur_epoch(
        handler,
        epoch,
        exp_id,
        threshold_configuration,
        mte,
        cur,
        new_vom_fnames,
        newly_fingerprinted_model_starts
    )

    # For re-evaluating in the past
    handler.all_models_for_epoch[ epoch ] = difference_info.all_models
    handler.new_models_for_epoch[ epoch ] = difference_info.new_models

    cur_vom = handler.combined_models_for_epoch[ epoch ]

    if difference_info.comparison_result_type != ModelComparisonInfo.ComparisonResultType.NOT_ENOUGH_DATA:
        log.info( "Doing epoch comparison against baseline. Epoch %d", epoch )
        baseline_similar, baseline_diff_info = do_vom_comparisons_against_baseline(
            handler,
            epoch,
            exp_id,
            prev_agg_metrics,
            cur_agg_metrics,
            cur_vom,
            threshold_configuration,
            difference_info.all_models,
            difference_info.new_models,
            cur
        )

        # Don't evaluate rules when we have no data
        if baseline_diff_info.comparison_result_type == ModelComparisonInfo.ComparisonResultType.MODELS_NOT_SIMILAR:
            log.info( "Got new, active models: %s %s", baseline_diff_info.new_models, baseline_diff_info.all_models )
            eval_result = mte.evaluate_all_rules( baseline_diff_info, prev_agg_metrics, cur_agg_metrics )
            if eval_result.rule_was_fired():
                cur.execute( "INSERT INTO rules_fired VALUES ( %s, %s, %s )", (exp_id, epoch, '\n'.join( eval_result.get_fired_rules() ) ) )

        log.info( "Outcome for epoch %d: %s", epoch, baseline_diff_info.comparison_result_type )
        if difference_info.comparison_result_type == ModelComparisonInfo.ComparisonResultType.CI_VIOLATING_MODELS:
            log.info( "GOT OTHER OUTCOME FOR EPOCH %d: %s", epoch, difference_info.comparison_result_type )
            eval_result = mte.evaluate_all_rules( difference_info, prev_agg_metrics, cur_agg_metrics )
            log.info( "Executed rule: %s", eval_result.rule_was_fired() )
            if eval_result.rule_was_fired():
                cur.execute( "INSERT INTO rules_fired VALUES ( %s, %s, %s )", (exp_id, epoch, '\n'.join( eval_result.get_fired_rules() ) ) )

    pg_commit( pg_conn, cur )

def start_dool( interval: int, csv_fname: str ) -> Any:
    try:
        os.remove( csv_fname )
    except:
        pass

    dool_proc = subprocess.Popen( ["dool", "--noupdate", "-cmndt", "--output", csv_fname,  str(interval) ], stdout=subprocess.DEVNULL )
    return dool_proc

def main_loop(
    sock: socket.socket,
    notifier: Any,
    handler: ModelDumpEventHandler,
    mousetrap_engine: MousetrapRuleEngine,
    pg_conn: Any,
    dool_proc: Any,
    exp_id: int,
    interval: int,
    cutoff: int,
    threshold_configuration: ThresholdConfiguration,
    dool_file: str
) -> None:

    cur_epoch = 0 # start 0
    try:
        while True:
            log.info( "Sleeping for {} seconds...".format( interval ) )
            time.sleep( interval )
            log.info( "Woke up after {} seconds...".format( interval ) )
            conn_alive = send_command( sock, Command.DUMP_AND_START_NEW_EPOCH )
            log.info( "Ended epoch %d at %s", cur_epoch, datetime.datetime.now() )

            t = threading.Timer(
                cutoff,
                contrast_models_and_respond,
                args=[
                    handler,
                    pg_conn,
                    mousetrap_engine,
                    exp_id,
                    cur_epoch,
                    threshold_configuration,
                    dool_file
                ]
            )
            t.start()

            cur_epoch += 1
            if not conn_alive:
                time.sleep( cutoff+1 )
                log.info( "Connection hung up. Shutting down." )
                break
    except KeyboardInterrupt:
        pass
    sock.close()
    notifier.stop()
    dool_proc.kill()
    for epoch in handler.combined_models_for_epoch.keys():
        vom = handler.combined_models_for_epoch[ epoch ]
        vom_fname = "/hdd1/dendrite_data/combined_model_for_epoch_{}.ser".format( epoch )
        with open( vom_fname, "wb" ) as f:
            f.write( vom.serialize() )
    #for fname in handler.file_models.keys():
        #max_epoch = max( handler.file_models[fname].keys() )
        #vom = handler.file_models[fname][max_epoch]
        #vom_fname = fname + "_arrival_estimator.ser"
        #with open( vom_fname, "wb" ) as f:
        #    f.write( vom.serialize() )


def load_properties( filepath: str, sep: str ='=', comment_char: str ='#' ) -> Dict[str, str]:
    """Read the file passed as parameter as a properties file."""
    props = {} # Dict[str, str]
    with open(filepath, "rt") as f:
        for line in f:
            l = line.strip()
            if l and not l.startswith(comment_char):
                key_value = l.split(sep)
                key = key_value[0].strip()
                value = sep.join(key_value[1:]).strip().strip('"')
                props[key] = value
    return props

def check_valid_rules(
    rules: List[str],
    tag_map: Dict[str,dendrite_analysis.FileLocation]
) -> None:
    for rule in rules:
        rr = RuleReader()
        try:
            pt = rr.parse_rule_action( rule )
            for tag in rr.get_tags():
                unquote_tag = tag[1:-1]
                if unquote_tag not in tag_map:
                    raise ParseFailedException( "Unknown tag: {}".format( tag ) )
        except ParseFailedException as e:
            log.error( "Could not parse rule: %s", rule )
            raise( e )

def load_rules(
    filepath: str,
    tag_map: Dict[str,dendrite_analysis.FileLocation]
) -> MousetrapRuleEngine:
    """Read the file passed as a parameter as a set of rules, one per line"""
    log.debug( "Loading rules from file %s", filepath )
    with open( filepath, "r" ) as f:
        rules = [ line.strip() for line in f ]
        log.debug( "Reading rules %s from file", rules )
        check_valid_rules( rules, tag_map )
        log.debug( "Rules are syntactically valid, registering them." )
        mte = MousetrapRuleEngine( rules, tag_map )
        return mte

def load_models( filename: str, handler: ModelDumpEventHandler ) -> None:
    props = load_properties( filename )
    for prop in props.keys():
        f_data = None
        with open( props[prop], "rb" ) as f:
            f_data = f.read()
        vom = dendrite_analysis.VariableOrderMarkovGraph.deserialize( f_data )
        handler.register_model( prop, vom )

def read_baseline_model( baseline_model_file: str ) -> dendrite_analysis.VariableOrderMarkovGraph:
    with open( baseline_model_file, "rb" ) as f:
        data = f.read()
        baseline_vom = dendrite_analysis.VariableOrderMarkovGraph.deserialize( data ) # type: dendrite_analysis.VariableOrderMarkovGraph
        return baseline_vom

def read_baseline_metrics( baseline_metrics_file: str ) -> dendrite_analysis.DstatFileData.DstatMetricData:
    with open( baseline_metrics_file, "r" ) as f:
        data = f.read()
        split_data = data.strip()[1:-1].split(",") # chop [ and ]
        csv_fields = [ float(f.lstrip()) for f in split_data[:13] ]
        ts = str( split_data[13].lstrip() )
        return dendrite_analysis.DstatFileData.DstatMetricData( csv_fields, ts )

def construct_rule_register_function( mte: MousetrapRuleEngine, tag_map: Dict[str, dendrite_analysis.FileLocation] ) -> Any:
    def register_func( rule: str ) -> None:
        try:
            check_valid_rules( [rule], tag_map )
            log.info( "Going to register function in mte {}".format( mte ) )
            mte.late_register_rule( rule )
        except:
            log.error( "Cannot register invalid rule, ignoring." )
    return register_func

def setup_cmd_client( args: Any ) -> Any:
    sock = connect_or_die( args.host, args.port )
    log.info( "Connected to data system." )
    baseline_vom = read_baseline_model( args.baseline_vom )
    baseline_metrics = read_baseline_metrics( args.baseline_metrics )
    pg_conn = psycopg2.connect( "host=tem05 user=postgres dbname={}".format( args.database_name ) )
    cur = pg_create_cursor( pg_conn )
    cur.execute( "INSERT INTO experiments DEFAULT VALUES RETURNING id" )
    exp_id = cur.fetchone()[0] # type: int
    pg_commit( pg_conn, cur )

    cur = pg_create_cursor( pg_conn )
    cur.execute( "SELECT fname, line, tag FROM tags" )
    rows = cur.fetchall()
    tag_map = {}
    for row in rows:
        tag_map[ row[2] ] = dendrite_analysis.FileLocation( row[0], int(row[1] ) )
    pg_commit( pg_conn, cur )

    threshold_configuration = ThresholdConfiguration( args.probability_threshold, args.frequency_threshold, args.fingerprint_threshold )
    notifier, handler = create_watchdog( args.t, args.learning_batch_size, baseline_vom, baseline_metrics, threshold_configuration, pg_conn, exp_id, tag_map )
    log.info( "Created watch dog." )
    load_models( args.config, handler )
    log.info( "Models loaded and registered." )
    mte = load_rules( args.rules, tag_map )
    log.info( "Rules loaded and registered." )
    dool_proc = start_dool( args.t, args.dool_file )

    rule_http_server = HTTPServer( ('', 8080), make_register_server_class( construct_rule_register_function( mte, tag_map ) ) )
    threading.Thread(target=rule_http_server.serve_forever).start()

    ret_dict = { "sock": sock,
        "notifier": notifier,
        "handler": handler,
        "mousetrap_engine": mte,
        "dool_proc": dool_proc,
        "pg_conn": pg_conn,
        "exp_id": exp_id
    }

    return ret_dict

if __name__ == "__main__":

    desc = "Command client to dump Sentinel models, compare them, and issue commands to instrumented system."
    parser = argparse.ArgumentParser( description=desc )
    parser.add_argument( "-t", metavar="interval", type=int, required=True,
        help="interval at which to dump models" )
    parser.add_argument( "--host", metavar="host", type=str, required=True,
        help="hostname of Sentinel command server" )
    parser.add_argument( "--port", metavar="port", type=int, required=True,
        help="port of Sentinel command server" )
    parser.add_argument( "--database-name", metavar="dbname", type=str, required=False, default="dendrite",
        help="PostgreSQL database on tem05 in which to store results." )

    parser.add_argument( "--cutoff", metavar="cutoff", type=int, required=False,
        default=5, help="cutoff for procs to dump models to be considered as part of behaviour change" )
    parser.add_argument( "--probability-threshold", metavar="threshold", type=float, required=False,
        default=2, help="Mean probability ratio difference in event counts to be considered a behaviour change." )
    parser.add_argument( "--frequency-threshold", metavar="threshold", type=float, required=False,
        default=2, help="Mean frequency ratio difference in event counts to be considered a behaviour change." )
    parser.add_argument( "--config", metavar="config", type=str, required=True,
        help="Config file with models from processes to identify what type of model is missing" )
    parser.add_argument( "--rules", metavar="rules", type=str, required=True,
        help="Rules file with executable actions" )
    parser.add_argument( "--baseline-vom", metavar="vom-file", type=str, required=True,
        help="Baseline combined VOM of representative behaviour" )
    parser.add_argument( "--baseline-metrics", metavar="metrics-line-file", type=str, required=True,
        help="Metrics for the baseline VOM at the epoch it was recorded" )
    parser.add_argument( "--dool-file", metavar="file", type=str, required=False, default="test.csv",
        help="File that dool will write metrics into." )
    parser.add_argument( "--fingerprint-threshold", metavar="threshold", type=float, required=False, default=3.,
        help="Mean probability difference for a model to be considered the same as a registered model.")
    parser.add_argument( "--learning-batch-size", metavar="batchsize", type=int, required=False, default=10,
        help="Interval at which to update model arrival time predictors." )

    args = parser.parse_args()

    config = setup_cmd_client( args )

    config['interval'] = args.t
    config['cutoff'] = args.cutoff
    threshold_configuration = ThresholdConfiguration( args.probability_threshold, args.frequency_threshold, args.fingerprint_threshold )
    config['threshold_configuration'] = threshold_configuration
    config['dool_file'] = args.dool_file

    main_loop( **config )
