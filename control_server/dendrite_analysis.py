import psycopg2 # type: ignore
import graphviz # type: ignore
import random
import numpy as np # type: ignore
import pyemd # type: ignore
import multiprocessing
import glob 
import pickle
import scipy.stats # type: ignore
import logging
import logging.config
import os
import math
import time
import datetime
import copy

from typing import List, Dict, Tuple, Set, Any, Iterable, Optional, Callable
from typing import no_type_check
from abc import ABC, abstractmethod
from colorama import Fore, Style # type: ignore

log = logging.getLogger( __name__ )

class FileBuffer:

    def __init__( self, file_handle: int ) -> None:
        self.prev_line_data = [] # type: List[str]
        self.cur_buffer = "" # type: str
        self.fh = file_handle

    def read_raw_data_from_file( self ) -> str:
        data_buff = ""
        while True:
            data = os.read( self.fh, 4096 )
            if data:
                data_buff += data.decode( "ascii" )
            else:
                break
        return data_buff

    def process_new_file_data( self ) -> bool:
        """ Read the file we are buffering and check if we have new data."""
        self.cur_buffer += self.read_raw_data_from_file()
        line_split_buffer = self.cur_buffer.split( "\n" )
        if len(line_split_buffer) == 1:
            # No newlines present, need to wait for more data
            return False

        # Add everything before the file newline chunk to lines
        # we know about.
        self.prev_line_data.extend( line_split_buffer[:-1] )
        self.cur_buffer = line_split_buffer[-1]
        return True

class DstatFileData:

    class DstatMetricData:
        
        def __init__( self, csv_fields: List[float], time: str ) -> None:
            self.cpu_usr = csv_fields[0] # type: float
            self.cpu_sys = csv_fields[1] # type: float
            self.cpu_idl = csv_fields[2] # type: float
            self.cpu_wai = csv_fields[3] # type: float
            self.cpu_stl = csv_fields[4] # type: float
            self.mem_used = csv_fields[5] # type: float
            self.mem_free = csv_fields[6] # type: float
            self.mem_buff = csv_fields[7] # type: float
            self.mem_cache = csv_fields[8] # type: float
            self.net_recv = csv_fields[9] # type: float
            self.net_send = csv_fields[10] # type: float
            self.disk_read = csv_fields[11] # type: float
            self.disk_write = csv_fields[12] # type: float
            self.time = time

        @staticmethod
        def get_field_index( metric_name: str ) -> int:
            if metric_name == 'cpu_usr':
                return 0
            elif metric_name == 'cpu_sys':
                return 1
            elif metric_name == 'cpu_idl':
                return 2
            elif metric_name == 'cpu_wai':
                return 3
            elif metric_name == 'cpu_stl':
                return 4
            elif metric_name == 'mem_used':
                return 5
            elif metric_name == 'mem_free':
                return 6
            elif metric_name == 'mem_buff':
                return 7
            elif metric_name == 'mem_cache':
                return 8
            elif metric_name == 'net_recv':
                return 9
            elif metric_name == 'net_send':
                return 10
            elif metric_name == 'disk_read':
                return 11
            elif metric_name == 'disk_write':
                return 12
            return -1

        def get_pg_tup( self ) -> Tuple[datetime.datetime,float,float,float,float,float,float,float,float,float,float,float,float,float]:
            dt = datetime.datetime.strptime( self.time, "%b-%d %H:%M:%S" )
            dt = dt.replace(year = datetime.datetime.now().year)
            return (dt, self.cpu_usr, self.cpu_sys, self.cpu_idl, self.cpu_wai, self.cpu_stl, self.mem_used, self.mem_free,
                self.mem_buff, self.mem_cache, self.net_recv, self.net_send, self.disk_read, self.disk_write)

        def get_agg_metric_value( self, metric_name: str ) -> float:
            if metric_name == "'cpu_usr'":
                return self.cpu_usr
            elif metric_name == "'cpu_sys'":
                return self.cpu_sys
            elif metric_name == "'cpu_idl'":
                return self.cpu_idl
            elif metric_name == "'cpu_wai'":
                return self.cpu_wai
            elif metric_name == "'cpu_stl'":
                return self.cpu_stl
            elif metric_name == "'mem_used'":
                return self.mem_used
            elif metric_name == "'mem_free'":
                return self.mem_free
            elif metric_name == "'mem_buff'":
                return self.mem_buff
            elif metric_name == "'mem_cache'":
                return self.mem_cache
            elif metric_name == "'net_recv'":
                return self.net_recv
            elif metric_name == "'net_send'":
                return self.net_send
            elif metric_name == "'disk_read'":
                return self.disk_read
            elif metric_name == "'disk_write'":
                return self.disk_write
            raise KeyError( "Metric not found: " + metric_name )

    def __init__( self, dstat_file_handle: int ) -> None:
        self.file_buffer = FileBuffer( dstat_file_handle )
        self.past_header = False

    def process_new_file_data( self ) -> bool:
        return self.file_buffer.process_new_file_data()

    def process_ready_data( self ) -> DstatMetricData:
        prev_line_data = self.file_buffer.prev_line_data
        assert len( prev_line_data ) >= 1
        if not self.past_header:
            assert len( prev_line_data ) > 1
            csv_fields = prev_line_data[-1].split( "," )
            # just cast a float to make sure this is a true line
            assert float(csv_fields[0]) > 0
            self.past_header = True
            log.debug( "dstat csv fields: %s", csv_fields )
        else:
            
            csv_fields = prev_line_data[-1].split( "," )
            assert float(csv_fields[0]) > 0
            log.debug( "dstat csv fields: %s", csv_fields )

        self.file_buffer.prev_line_data.clear()
        data = DstatFileData.DstatMetricData( [ float(c) for c in csv_fields[:-1] ], csv_fields[-1] )
        return data

class EpochFileData( ABC ):
    """A buffer of data read from a file that represents a VariableOrderMarkovGraph (vom)."""

    def __init__( self, file_handle: int ) -> None:
        self.file_buffer = FileBuffer( file_handle )
        self.model_ready = False # type: bool

    def process_new_file_data( self ) -> bool:
        """Add the data string from the file to this buffer. """

        have_new_file_data = self.file_buffer.process_new_file_data()
        if have_new_file_data:
            for line in self.file_buffer.prev_line_data:
                if line.startswith( "." ):
                    self.model_ready = True
                    return True
        return False

    def is_ready( self ) -> bool:
        """Is the model ready for processing (we have hit a .<N> epoch marker indicating
        that the full model was written)?"""
        return self.model_ready

    @abstractmethod
    def process_ready_data( self ) -> Any:
        pass

class MetricsFileData( EpochFileData ):
    """Metrics Data from a file."""

    class MetricsTransitionKey:
        """A key that identifies a particular transition"""
        def __init__( self, prior_event_seq: List[int], next_event: int ):
            self.prior_event_seq = prior_event_seq
            self.next_event = next_event

        def __eq__( self, obj: Any ) -> bool:
            return isinstance( obj, MetricsFileData.MetricsTransitionKey ) and self.prior_event_seq == obj.prior_event_seq and self.next_event == obj.next_event

        def __ne__( self, obj: Any ) -> bool:
            return not self == obj

        def __hash__( self ) -> int:
            return hash( (str(self.prior_event_seq), self.next_event) )

        def __repr__( self ) -> str:
            return str(self.prior_event_seq) + " -> " + str(self.next_event)

    class MetricsData:
        """Samples of a particular metric. Samples are bounded in size (usually 1000 samples total)
        with a total count of count"""
        def __init__( self, count: int, sampled_vals: List[float] ) -> None:
            self.count = count
            self.sampled_vals = sampled_vals
        def __repr__( self ) -> str:
            return "( {}, {} )".format( self.count, self.sampled_vals )

    @staticmethod
    def read_metrics_data_from_lines( lines: List[str] ) -> Tuple[int, List[Tuple[List[int],int]], Dict['MetricsFileData.MetricsTransitionKey', Dict[str,'MetricsFileData.MetricsData']]]:
        i = 0
        cur_epoch_transitions = [] # type: List[ Tuple[ List[int], int ]]
        cur_mt_key = None
        transition_metrics = {} # type: Dict[MetricsFileData.MetricsTransitionKey, Dict[str,MetricsFileData.MetricsData]]

        for line in lines:
            if line.startswith( "." ):
                break
            if "->" in line:
                seq_str, dst_str = line.split("->")
                dst = int(dst_str.strip()[:-1])
                event_seq_ids_str = seq_str[1:-2].split(",")
                event_seq_ids = [ int( k.strip() ) for k in event_seq_ids_str ]
                cur_mt_key = MetricsFileData.MetricsTransitionKey( event_seq_ids, dst )
                cur_epoch_transitions.append( ( event_seq_ids, dst ) )
                i += 1
                continue

            metric_name, metric_data = line.split(":") 
            metric_count = 0
            metric_vals = [] # type: List[float]
            if not "empty" in metric_data:
                metric_str_count, metric_str_vals = metric_data.split(";")
                metric_count = int(metric_str_count)
                metric_str_vals_list = metric_str_vals.split(",")[:-1]
                metric_vals = [ float(val) for val in metric_str_vals_list ]

            md = MetricsFileData.MetricsData( metric_count, metric_vals )
            assert cur_mt_key is not None
            if not cur_mt_key in transition_metrics:
                transition_metrics[ cur_mt_key ] = {}
            transition_metrics[ cur_mt_key ][ metric_name ] = md
            
            i += 1 


        return (i, cur_epoch_transitions, transition_metrics)


    def process_ready_data( self ) -> Tuple[ int, List[Tuple[ List[int], int ]], Dict['MetricsFileData.MetricsTransitionKey', Dict[str,'MetricsFileData.MetricsData']]]:

        
        i, cur_epoch_transitions, transition_metrics = MetricsFileData.read_metrics_data_from_lines( self.file_buffer.prev_line_data )
        epoch = self.file_buffer.prev_line_data[i][1:].strip() # ".<N>\n"
        epoch_int = int(epoch)

        self.file_buffer.prev_line_data = self.file_buffer.prev_line_data[i+1:]

        return (epoch_int, cur_epoch_transitions, transition_metrics)

class ModelFileData( EpochFileData ):
    """Model data for a particular process."""

    def __init__( self, model_file_handle: int, metrics_file_handle: int ) -> None:
        super().__init__( model_file_handle )
        self.metrics_file_data = MetricsFileData( metrics_file_handle )

    def process_ready_data( self ) -> 'VariableOrderMarkovGraph':
        """Given the buffered data for a model that is_ready for processing, convert it
        into a VariableOrderMarkovGraph."""
        assert self.model_ready

        self.metrics_file_data.process_new_file_data()
        assert self.metrics_file_data.is_ready()

        i = 0
        for line in self.file_buffer.prev_line_data:
            if line.startswith( "." ):
                break
            i += 1
                
        try:
            vom = process_dump_lines( self.file_buffer.prev_line_data[:i] )

        except Exception as e:
            raise e

        # skip epoch
        epoch = self.file_buffer.prev_line_data[i][1:].strip() # ".<N>\n"
        epoch_int = int(epoch)

        # Right here I would incorporate the model data.
        metrics_epoch_int, cur_epoch_transitions, metrics_for_transitions = self.metrics_file_data.process_ready_data()
        assert epoch_int == metrics_epoch_int

        vom.set_epoch( epoch_int )
        vom.incorporate_transition_metrics( metrics_for_transitions )
        
        self.file_buffer.prev_line_data = self.file_buffer.prev_line_data[i+1:]

        # Reset model ready
        self.model_ready = False
        for line in self.file_buffer.prev_line_data:
            if line.startswith( "." ):
                self.model_ready = True
                break
        return vom

class FileLocation:
    """A location in a file for a given event (filename,line_number)"""
    def __init__( self, fname: str, line_number: int ):
        self.fname = fname
        self.line_number = line_number
    def __repr__( self ) -> str:
        return "{}:{}".format( self.fname, self.line_number )
    def __eq__( self, obj: Any ) -> bool:
        return isinstance( obj, FileLocation ) and self.line_number == obj.line_number and self.fname == obj.fname
    def __ne__( self, obj: Any ) -> bool:
        return not self == obj
    def __hash__( self ) -> int:
        return hash( (self.fname, self.line_number) )

class EventRecord:
    def __init__( self, fname: str , ln: int, count: int, prob: float  ) -> None:
        self.event_loc = FileLocation( fname, ln )
        self.count = count
        self.prob =  prob

    def get_id( self ) -> str:
        return self.event_loc.__repr__()

    def __repr__(self) -> str:
        return "<{}, Count: {},Prob: {}>".format( self.get_id(), self.count, self.prob )

class TransitionRecord:
    """ A transition record between markov nodes. node == dest."""
    def __init__( self, dst: 'MarkovNode', prob: float, transition_time_cdf: List[Tuple[float,float]] ):
        self.dst = dst
        self.prob = prob
        self.transition_time_cdf = transition_time_cdf
        self.good_path = False
    def is_good_path( self ) -> bool:
        return self.good_path
    def set_good_path( self ) -> None:
        self.good_path = True
    def set_bad_path( self) -> None:
        self.good_path = False

class MarkovNode:
    """A Node in a MarkovGraph"""
    def __init__( self, node_id: int, fname: str, line: int ) -> None:
        self.node_id = node_id
        self.transitions = [] # type: List[TransitionRecord]
        self.event_loc = FileLocation( fname, line )
    
    def add_transition( self, dst: 'MarkovNode', prob: float, transition_time_cdf: List[Tuple[float,float]]) -> None:
        """ Add a transition from this node to dst with probability prob and transition_time_cdf"""
        tr = TransitionRecord( dst, prob, transition_time_cdf )
        self.transitions.append( tr )

    def get_transition( self, dst_name: str, dst_line: int ) -> 'MarkovNode':
        for transition in self.transitions:
            if transition.dst.event_loc.fname == dst_name and transition.dst.event_loc.line_number == dst_line:
                return transition.dst
        raise KeyError( "No such transition found: {}:{} from {}:{}".format( dst_name, dst_line, self.event_loc.fname, self.event_loc.line_number ) )
    
    def sample_transition_time( self, dst: 'MarkovNode' ) -> float:
        """ Sample a transition time from the current node to dst"""
        for tr in self.transitions:
            if dst != tr.dst:
                continue
            cdf_draw = random.random()
            if not tr.transition_time_cdf:
                print( "Unknown Transition!")
                return 0.0
            for pctl, val in tr.transition_time_cdf:
                if cdf_draw <= pctl:
                    return val
            return tr.transition_time_cdf[-1][1]
        raise KeyError( "Could not find transition for dst: {}".format( dst ) )
    
    def __str__( self ) -> str:
        return self.__repr__()
    
    def __repr__( self ) -> str:
        return "MarkovNode-{}".format( self.node_id )

    def get_key( self ) -> str:
        return self.event_loc.__repr__()
    
class MarkovGraph:
    """A graph of MarkovNodes"""
    def __init__( self, nodes: List[MarkovNode] ) -> None:
        self.node_map = {} # type: Dict[str, MarkovNode]
        for node in nodes:
            self.node_map[node.get_key()] = node

    def get_node( self, fname: str, line: int ) -> MarkovNode:
        return self.node_map[ FileLocation( fname, line ).__repr__() ]

    def __repr__( self ) -> str:
        return str(self.node_map)

@no_type_check
def get_log_transitions( pg_conn: Any, src_fname: str, src_line: int, run_id: int ) -> List[Tuple[str,int,float]]:
    """Get the transitions from event src_fname:src_line in run_id using the provided postgres connection"""
    cur = pg_conn.cursor()
    get_transition_probs_query = "SELECT log_next_fname, log_next_line, transition_probability FROM log_line_transitions WHERE log_initial_fname = %s AND log_initial_line = %s AND run_id = %s"

    cur.execute( get_transition_probs_query, ( src_fname, src_line, run_id ) )
    results = cur.fetchall() # type: List[Tuple[str,int,float]]
    cur.close()
    return results

@no_type_check
def get_log_transition_time( pg_conn: Any, src_fname: str, src_line: int, dst_fname: str, dst_line: int, run_id: int ) -> List[Tuple[List[int], List[float]]]:
    """Get the transition times from event src_fname:src_line to dst_fname:dst_line in run_id using the provided postgres connection"""
    cur = pg_conn.cursor()
    get_cdf_query = "SELECT percentiles, percentile_values FROM transition_cdfs WHERE src_fname = %s and src_line = %s AND dst_fname = %s and dst_line = %s and run_id = %s"
    
    cur.execute( get_cdf_query, ( src_fname, src_line, dst_fname, dst_line, run_id ) )
    results = cur.fetchall() # type: List[Tuple[List[int], List[float]]]
    cur.close()
    return results

@no_type_check
def build_full_markov_graph( pg_conn: Any, run_id: int ) -> MarkovGraph:
    """ Build a markov graph for run_id using the provided postgres connection"""
    get_all_nodes_query = """SELECT log_initial_fname, log_initial_line FROM log_line_transitions WHERE run_id = %s UNION SELECT log_next_fname, log_next_line FROM log_line_transitions WHERE run_id = %s"""
    
    nodes = [] # type: List[MarkovNode]
    cur = pg_conn.cursor()
    cur.execute( get_all_nodes_query, ( run_id, run_id ) )
    data = cur.fetchall() # type: List[Tuple[str,int]]
    cur.close()
    for entry in data:
        node = MarkovNode( hash(entry[0]) ^ hash(entry[1]), entry[0], entry[1] )
        nodes.append( node )
    
    for node in nodes:
        src_fname = node.event_loc.fname
        src_line = node.event_loc.line_number
        transitions = get_log_transitions( pg_conn, src_fname, src_line, run_id )
        for dst_name, dst_line, dst_prob in transitions:
            dst_node = node.get_transition( dst_name, dst_line )
            cdf_data = get_log_transition_time( pg_conn, node.event_loc.fname, node.event_loc.line_number, dst_node.event_loc.fname, dst_node.event_loc.line_number, run_id )
            if len(cdf_data) < 1:
                raise KeyError( "No transition for: {}:{} -> {}:{}".format( node.event_loc.fname, node.event_loc.line_number, dst_node.event_loc.fname, dst_node.event_loc.line_number) )
            else:
                node.add_transition( dst_node, dst_prob, list(zip(cdf_data[0][0],cdf_data[0][1])) )
            
    return MarkovGraph(nodes)

def bounded_dfs( node: MarkovNode, goal_nodes: List[MarkovNode], cur_prob: float =1., cut_off: float = 1E-5, 
    nodes_seen_so_far: List[MarkovNode] = [], allow_loops: bool =True ) -> bool:

    if node in goal_nodes:
        return True
    
    if not allow_loops and node in nodes_seen_so_far:
        return False
    
    is_good_path = False
    
    for transition in node.transitions:
        
        next_prob = cur_prob * transition.prob
        if next_prob >= cut_off:
            result = bounded_dfs( transition.dst, goal_nodes, next_prob, cut_off, nodes_seen_so_far + [ node ], allow_loops )
            if result:
                # Mark the transition as a known good transition
                is_good_path = True
                transition.set_good_path()
            else:
                transition.set_bad_path()
    return is_good_path

def get_transition_node( valid_transitions: List[TransitionRecord] ) -> MarkovNode:
    """Given a set of valid transitions, determine which transition we will walk next
    according to their normalized probabilities."""
    prob = random.random()
    normalization_factor = 0.0
    for transition in valid_transitions:
        normalization_factor += transition.prob

    assert( normalization_factor <= 1.1 ) # to handle floating point error

    for transition in valid_transitions:
        adj_prob = transition.prob / normalization_factor
        if prob <= adj_prob:
            return transition.dst
        prob -= adj_prob
    raise ArithmeticError( "Ran out of normalized transition probabilities to walk! Normalization error?" )

def bounded_random_walk( start_node: MarkovNode, terminal_nodes: List[MarkovNode], 
    num_walks: float =1.E6 ) -> Tuple[Dict[MarkovNode,int],Dict[MarkovNode,List[float]]]:
    """ Conduct num_walks bound random walks from the start_node to one of the terminal_nodes. Normalizes probabilities
    by pruning away paths that do not reach terminal nodes. Returns a tuple where the first item is a dictionary of the number times we've hit the terminal
    nodes and a dictionary representing the elapsed durations of how long it took to hit those terminal nodes (determined by MCMC)"""

    # Mark up the transitions so we have "railings" and know where we can go during our walk
    bounded_dfs( start_node, terminal_nodes, allow_loops=False )
    
    cur_walk = 0
    terminal_node_count = {} # type: Dict[MarkovNode, int]
    node_elapsed_times = {} # type: Dict[MarkovNode, List[float]]
    for tn in terminal_nodes:
        terminal_node_count[tn] = 0
        node_elapsed_times[tn] = []
        
    while cur_walk < num_walks:
        cur_node = start_node
        elapsed_ms = 0.
        while True:
            # TODO: prune this run out if the probably goes below a certain threshold --- its effectively zero
            # so might as well shortcut the results.
            # Problem is if the thing we prune out is a massive heavy hitter on run_time --- could inflate average!
            if cur_node in terminal_nodes:
                cur_walk += 1
                terminal_node_count[cur_node] += 1
                node_elapsed_times[cur_node].append( elapsed_ms )
                break
            # Get all the transitions we can take out of here
            valid_transitions = [ transition for transition in cur_node.transitions if transition.is_good_path() ]
            
            next_node = get_transition_node( valid_transitions )

            sampled_transition_time = cur_node.sample_transition_time( next_node )
            elapsed_ms += sampled_transition_time
            cur_node = next_node

    return terminal_node_count, node_elapsed_times

def depth_bounded_mcmc( start_node: MarkovNode, target_depth: int ) -> Tuple[Dict[MarkovNode, int], Dict[MarkovNode, List[float]]]:
    """Do bounded random walks MCMC with terminal nodes equal to the set of all possible nodes at depth k.
    If these nodes are also available at a higher depth, may terminate earlier (e.g. loops)"""
    def get_terminal_nodes_at_k_depth( cur_node: MarkovNode, nodes: Set[MarkovNode], cur_depth: int, target_depth: int ) -> None:
        if cur_depth == target_depth:
            nodes.add( cur_node )
            return
        next_nodes = [ t.dst for t in cur_node.transitions ] # type: List[MarkovNode]
        for nn in next_nodes:
            get_terminal_nodes_at_k_depth( nn, nodes, cur_depth+1, target_depth )

    nodes = set() # type: Set[MarkovNode]
    cur_depth = 0
    get_terminal_nodes_at_k_depth( start_node, nodes, 0, target_depth )
    counts, elapsed_time = bounded_random_walk( start_node, list(nodes) )
    return counts, elapsed_time

def compute_percentiles_from_mcmc_results( timer_results: Dict[MarkovNode, List[float]] ) -> Dict[MarkovNode, List[float]]:
    """ Compute percentiles from a dictionary of terminal nodes to elapsed times"""
    percentiles = np.arange( 5,100, 5 )
    percentile_results = {} # type: Dict[MarkovNode, List[float]]
    for node, times in timer_results.items():
        times_array = np.array( times )
        percentile_results[ node ] = np.percentile( times_array, percentiles )
    return percentile_results

@no_type_check
def get_data_from_postgres( conn: Any, run_id: int ) -> Tuple[Dict[str,EventRecord], Dict[str,Dict[str,float]]]:
    sql_stmt = "SELECT log_fname, log_line, log_count, log_probability FROM log_line_probabilities WHERE run_id = %s"
    cur = conn.cursor()
    cur.execute( sql_stmt, (run_id,) )
    rs = cur.fetchall()
    events = {} # type: Dict[str, EventRecord]
    for row in rs:
        fname, ln, count, prob = row
        event = EventRecord( fname, ln, count, prob )
        events[event.get_id()] = event

    sql_stmt = "SELECT log_initial_fname,log_initial_line, log_next_fname, log_next_line, transition_probability FROM log_line_transitions WHERE run_id = %s"
    cur = conn.cursor()
    cur.execute( sql_stmt, (run_id,) )
    rs = cur.fetchall()
    event_transitions = {} # type: Dict[str,Dict[str,float]]
    for row in rs:
        initial_fname, initial_line, next_fname, next_line, prob = row
        from_event_id = "{}:{}".format( initial_fname, initial_line )
        to_event_id = "{}:{}".format( next_fname, next_line )
        if not from_event_id in event_transitions:
            event_transitions[from_event_id] = {}
        event_transitions[from_event_id][to_event_id] = prob

    return events, event_transitions

class EventDifferenceRecord:
    def __init__( self, prob_diff_val: float, count_diff_val: float,
            left_prob: float, right_prob: float,
            left_count: int, right_count: int, event_fname: str, 
            event_line: int, is_left_greater: bool ) -> None:

        self.prob_diff_val = prob_diff_val
        self.count_diff_val = count_diff_val
        self.left_prob = left_prob
        self.right_prob = right_prob
        self.left_count = left_count
        self.right_count = right_count
        self.event_fname = event_fname
        self.event_line = event_line
        self.is_left_greater = is_left_greater

class TransitionDiffRecord:
    def __init__( self, diff_val: float, left_val: float, right_val: float, src_fname: str, 
            src_line: int, dst_fname: str, dst_line: int, is_left_greater: bool ) -> None:
        self.diff = diff_val
        self.left_val = left_val
        self.right_val = right_val
        self.src_fname = src_fname
        self.src_line = src_line
        self.dst_fname = dst_fname
        self.dst_line = dst_line
        self.is_left_greater = is_left_greater

def create_event_difference_record( prob_diff_val: float, count_diff_val: float,
        left_prob: float, right_prob: float,
        left_count: int, right_count: int, event_fname: str,
        event_line: int, is_left_greater: bool) -> EventDifferenceRecord:
    pdr = EventDifferenceRecord(
        prob_diff_val, count_diff_val, left_prob, right_prob, left_count, right_count,
        event_fname, event_line, is_left_greater )
    return pdr

def create_transition_diff_record( diff: float, left: float, right: float, event_1_fname: str,
        event_1_line: int, event_2_fname: str, event_2_line: int, is_left_greater: bool ) -> TransitionDiffRecord:
    tdr = TransitionDiffRecord( diff, left, right, event_1_fname, event_1_line, event_2_fname, event_2_line, is_left_greater )
    return tdr

def compute_event_difference_stats( event_difference_records: List[EventDifferenceRecord] ) -> Tuple[float,float]:
    """Given a list of EventDifferenceRecord from a VOM comparison, produce a
    Tuple( tot_diff, mean_diff, variance_diff )."""

    # Convert to array of diff scores.
    prob_diff_scores = np.array( [ event_difference_record.prob_diff_val for event_difference_record in event_difference_records ] )
    count_diff_scores = np.array( [ event_difference_record.count_diff_val for event_difference_record in event_difference_records ] )
    print( "Doing means", flush=True )
    mean_prob_diff = np.mean( prob_diff_scores )
    mean_count_diff = np.mean( count_diff_scores )
    print( "Done means", flush=True )
    assert event_difference_records

    return mean_prob_diff, mean_count_diff

def do_calc_(
    k: FileLocation,
    events_1: Dict[FileLocation, EventRecord],
    events_2: Dict[FileLocation, EventRecord],
    level: int,
    missing_events_level_score_mapper: Callable[[FileLocation, int, int, Dict[FileLocation, EventRecord], Dict[FileLocation, EventRecord]],float],
    log_level_score_rescale_func: Callable[[FileLocation, int, float, Dict[FileLocation, EventRecord], Dict[FileLocation, EventRecord]], float]
) -> EventDifferenceRecord:

    # Compute the event probabilities for both runs
    event1_prob = 0.0
    event2_prob = 0.0
    event1_count = 0
    event2_count = 0
    event_fname = None
    event_ln = None
    if k in events_1:
        event1_prob = events_1[k].prob
        event_fname = events_1[k].event_loc.fname
        event_ln = events_1[k].event_loc.line_number
        event1_count = events_1[k].count
    if k in events_2:
        event2_prob = events_2[k].prob
        event_fname = events_2[k].event_loc.fname
        event_ln = events_2[k].event_loc.line_number
        event2_count = events_2[k].count
    assert event_fname is not None
    assert event_ln is not None

    min_prob = min( event1_prob, event2_prob )
    max_prob = max( event1_prob, event2_prob )
    min_count = min( event1_count, event2_count )
    max_count = max( event1_count, event2_count )

    is_left_greater = True if event1_prob > event2_prob else False

    # If both sides are greater than zero, it is straightforward to calculate the difference b/c there
    # are no divisions by zero.
    if min_prob > 0:
        assert min_count > 0
        ratio_diff = max_prob / min_prob
        count_diff = max_count / min_count

        # Get level
        ratio_diff = log_level_score_rescale_func( k, level, ratio_diff, events_1, events_2 )
        count_diff = log_level_score_rescale_func( k, level, count_diff, events_1, events_2 )
        return create_event_difference_record( ratio_diff, count_diff, event1_prob, event2_prob, event1_count, event2_count, event_fname, event_ln, is_left_greater )
    else:
        assert min_count == 0

        ratio_diff = missing_events_level_score_mapper( k, level, max_count, events_1, events_2 )
        count_diff = ratio_diff

        return create_event_difference_record( ratio_diff, count_diff, event1_prob, event2_prob, event1_count, event2_count, event_fname, event_ln, is_left_greater )

def compute_event_difference(
    events_1: Dict[FileLocation, EventRecord],
    events_2: Dict[FileLocation, EventRecord],
    combined_event_level_map: Dict[FileLocation, int],
    missing_events_level_score_mapper: Callable[[FileLocation, int, int, Dict[FileLocation, EventRecord], Dict[FileLocation, EventRecord]],float],
    log_level_score_rescale_func: Callable[[FileLocation, int, float, Dict[FileLocation, EventRecord], Dict[FileLocation, EventRecord]],float]
) -> List[EventDifferenceRecord]:

    s = [] # type: List[EventDifferenceRecord]
    for k in set(events_1.keys()).union(events_2.keys()):
        level = combined_event_level_map[k] if k in combined_event_level_map else 20
        e_diff = do_calc_( k, events_1, events_2, level, missing_events_level_score_mapper, log_level_score_rescale_func )
        s.append( e_diff )
    s.sort(key=lambda diff_record: diff_record.prob_diff_val, reverse=True)
    return s

def compute_transition_diff( events_1: Dict[int, EventRecord], events_2: Dict[int, EventRecord], event_transitions1: Dict[int, Dict[int, float]], event_transitions2: Dict[int, Dict[int,float]] ) -> Tuple[float, List[TransitionDiffRecord], List[TransitionDiffRecord]]:
    agg_score = 0.
    score_diffs = []
    raw_transition_diffs = []
    for k in set(events_1).union(events_2):
        for k2 in set(events_1).union(events_2):
            events_1_prob = 0.0
            events_1_trans_prob = 0.0
            events_2_prob = 0.0
            events_2_trans_prob = 0.0
            if k in events_1:
                events_1_prob = events_1[k].prob
                if k in event_transitions1 and k2 in event_transitions1[k]:
                    events_1_trans_prob = event_transitions1[k][k2]
            if k in events_2:
                events_2_prob = events_2[k].prob
                if k in event_transitions2 and k2 in event_transitions2[k]:
                    events_2_trans_prob = event_transitions2[k][k2]
            events_1_score = events_1_prob * events_1_trans_prob
            events_2_score = events_2_prob * events_2_trans_prob
            is_left_greater = True if events_1_score > events_2_score else False
            difference = (events_1_score - events_2_score) ** 2
            agg_score += difference
            min_score = min(events_1_score, events_2_score)
            max_score = max(events_1_score, events_2_score)
            min_transition_score = min(events_1_trans_prob, events_2_trans_prob)
            max_transition_score = max(events_1_trans_prob, events_2_trans_prob)
            if min_score == 0.0:
                ratio_diff = float('Inf')
            else:
                ratio_diff = max_score / min_score
            if min_transition_score == 0.0:
                trans_ratio_diff = float('Inf')
            else:
                trans_ratio_diff = max_transition_score / min_transition_score

            fname_1 = events_1[k].event_loc.fname if k in events_1 else events_2[k].event_loc.fname
            line_1 = events_1[k].event_loc.line_number if k in events_1 else events_2[k].event_loc.line_number
            fname_2 = events_1[k2].event_loc.fname if k2 in events_1 else events_2[k2].event_loc.fname
            line_2 = events_1[k2].event_loc.line_number if k2 in events_1 else events_2[k2].event_loc.line_number

            score_diffs.append( create_transition_diff_record( ratio_diff, events_1_score, events_2_score, fname_1, line_1,
                                                              fname_2, line_2, is_left_greater ) )
            raw_transition_diffs.append( create_transition_diff_record( trans_ratio_diff, events_1_trans_prob, events_2_trans_prob,
                                                                       fname_1, line_1, fname_2, line_2, is_left_greater ))
    score_diffs.sort(key=lambda diff_record: diff_record.diff, reverse=True)
    raw_transition_diffs.sort(key=lambda diff_record: diff_record.diff, reverse=True)
    return agg_score, score_diffs, raw_transition_diffs

def pretty_print_differences( agg_score: float, s: List[EventDifferenceRecord], raw_transition_diffs: List[TransitionDiffRecord], score_diffs: List[TransitionDiffRecord], k: int =30) -> None:
    print( "Aggregate Difference: {}".format( agg_score ) )
    print( "="*60 )
    print( "Top {} Event Probability Differences:".format(k) )
    print( "{:<15}\t{:<15}\t{:<15}{:<20}".format("Ratio","Left Prob","Right Prob","Location") )
    print( "-"*60 )
    i = 0
    while i < k:
        if i == len(s):
            break
        if s[i].is_left_greater:
            print( Fore.BLUE + "{:<15f}\t{:<15f}\t{:<15f}\t{:<20}".format( 
                s[i].prob_diff_val, s[i].left_prob, s[i].right_prob,
                "{}:{}".format(s[i].event_fname, s[i].event_line)) + Style.RESET_ALL)
        else:
            print( Fore.RED + "{:<15f}\t{:<15f}\t{:<15f}\t{:<20}".format(
                s[i].prob_diff_val, s[i].left_prob, s[i].right_prob,
                "{}:{}".format(s[i].event_fname, s[i].event_line)) + Style.RESET_ALL)
        i += 1
    print( "="*60 )
    print( "Top {} Event Transition Differences:".format(k) )
    print( "{:<15}\t{:<15}\t{:<15}\t{:<20}".format("Ratio","Left Prob","Right Prob","Transition" ) )
    print( "-"*60 )
    i = 0
    while i < k:
        if i == len(raw_transition_diffs):
            break

        if raw_transition_diffs[i].is_left_greater:
            print( Fore.BLUE + "{:<15}\t{:<15}\t{:<15}\t{:<20}".format( raw_transition_diffs[i].diff, raw_transition_diffs[i].left_val,
                raw_transition_diffs[i].right_val, "{}:{} -> {}:{}".format( raw_transition_diffs[i].src_fname, raw_transition_diffs[i].src_line,
                raw_transition_diffs[i].dst_fname, raw_transition_diffs[i].dst_line)) + Style.RESET_ALL )
        else:
            print( Fore.RED + "{:<15}\t{:<15}\t{:<15}\t{:<20}".format( raw_transition_diffs[i].diff, raw_transition_diffs[i].left_val,
                raw_transition_diffs[i].right_val, "{}:{} -> {}:{}".format( raw_transition_diffs[i].src_fname, raw_transition_diffs[i].src_line, 
                raw_transition_diffs[i].dst_fname, raw_transition_diffs[i].dst_line)) + Style.RESET_ALL )
        i += 1

def show_transition_graph( start_node: MarkovNode, depth: int =1, filter_function: Callable[[MarkovNode,float],bool]=lambda node, prob: True ) -> Any:
    def remap_event_id( event_id: str ) -> str:
        return event_id.replace(":", "-")
    
    def build_transition_graph( graph: graphviz.Digraph, node: MarkovNode, depth: int, nodes_so_far: List[MarkovNode] ) -> None:
        if node in nodes_so_far:
            return
        
        nodes_so_far.append( node )
        graph.node( remap_event_id( node.get_key() ), node.get_key() )

        if depth != 0:
            for transition in node.transitions:
                if filter_function( transition.dst, transition.prob):
                    graph.node( remap_event_id( transition.dst.get_key() ), transition.dst.get_key() )
                    edge_label = "{:<5f}".format( transition.prob )
                    graph.edge( remap_event_id( node.get_key() ), remap_event_id( transition.dst.get_key() ), label=edge_label )
                    build_transition_graph( graph, transition.dst, depth-1, nodes_so_far )
    graph = graphviz.Digraph(comment="{} Event Transitions, Depth={}".format( start_node.get_key(), depth ) )
    build_transition_graph( graph, start_node, depth, [] )
    return graph

@no_type_check
def show_mcmc_graph( start_event_id, events, event_transitions, target_depth, depth_bounded_mcmc_results ):
    def remap_event_id( event_id ):
        return event_id.replace(":", "-")

    def build_transition_graph( graph, event_id, events, event_transitions, mcmc_results, depth ):
        event = events[event_id]
        graph.node( remap_event_id( event_id ), event_id )
        transitions = event_transitions[ event_id ]
        if depth == 0:
            return
        for transition in transitions:
            graph.node( remap_event_id( transition ), transition )
            # 50th percentile
            edge_label = "{:<5f}".format( mcmc_results[ event_id ][ transition ][ 9 ] )
            graph.edge( remap_event_id( event_id ), remap_event_id( transition ), label=edge_label )
            build_transition_graph( graph, transition, events, event_transitions, mcmc_results, depth-1 )

    graph = graphviz.Digraph(comment="{} Event MCMC, Depth={}".format( start_event_id, target_depth ) )

    mcmc_dict = {}
    for key in depth_bounded_mcmc_results:
        ptls = depth_bounded_mcmc_results[ key ]
        dst_id = key.get_key()
        if not start_event_id in mcmc_dict:
            mcmc_dict[ start_event_id ] = {}
        mcmc_dict[ start_event_id ][ dst_id ] = ptls
    build_transition_graph( graph, start_event_id, events, event_transitions, mcmc_dict, target_depth )
    return graph

@no_type_check
def generate_distance_matrix( percentile_vec ):
    """Compute the pairwise distance between every set of percentile "positions". Since the percentiles
    are not equally spread, some may be farther apart than others, which corresponds to more "distance" in terms of EMD"""
    grid1, grid2 = np.meshgrid( percentile_vec, percentile_vec )
    return np.abs( grid2 - grid1 )

@no_type_check
def get_cdf( conn, src_fname: str, src_line: int, dst_fname: str, dst_line: int, run_id: int ):
    query = "SELECT percentile_values FROM transition_cdfs WHERE run_id = %s AND src_fname = %s AND src_line = %s AND dst_fname = %s AND dst_line = %s"
    cur = conn.cursor()
    cur.execute( query, (run_id, src_fname, src_line, dst_fname, dst_line) )
    results = cur.fetchall()
    return results

@no_type_check
def do_emd( args ):
    src_fname, src_line, dst_fname, dst_line, cdf_vals1, cdf_vals2, dist_mat, normalize = args
    
    # Renormalize
    if normalize:
        max_val = max( cdf_vals1[-1], cdf_vals2[-1])
    else:
        max_val = 1.0
    cdf_vals1 = np.array( cdf_vals1 )/max_val
    cdf_vals2 = np.array( cdf_vals2 )/max_val
    
    # EMD
    emd_score = pyemd.emd( cdf_vals1, cdf_vals2, dist_mat )
    return (emd_score, src_fname, src_line, dst_fname, dst_line )

@no_type_check
def get_emd_scores_for_transitions( conn, run_id1: int, run_id2: int, normalize=True, procs=1 ):
    percentiles = [0.05,
         0.1,
         0.15,
         0.2,
         0.25,
         0.3,
         0.35,
         0.4,
         0.45,
         0.5,
         0.55,
         0.6,
         0.65,
         0.7,
         0.75,
         0.8,
         0.85,
         0.9,
         0.95,
         0.99,
         0.999 ]
    dist_mat = generate_distance_matrix( percentiles )
    transition_cdf_query = """SELECT DISTINCT(src_fname, src_line, dst_fname, dst_line) FROM transition_cdfs,
        log_line_transitions, log_line_probabilities
        WHERE src_fname = log_initial_fname and src_line = log_initial_line and dst_fname = log_next_fname and dst_line = log_next_line
        and log_line_transitions.run_id = transition_cdfs.run_id and log_line_transitions.run_id = log_line_probabilities.run_id and 
        log_fname = src_fname and log_line = src_line and transition_count > 1000 and 
        (transition_cdfs.run_id = %s OR transition_cdfs.run_id = %s)"""
    cur = conn.cursor()
    cur.execute( transition_cdf_query, (run_id1, run_id2) )
    results = cur.fetchall()
    cur.close()

    proc_pool = multiprocessing.Pool( procs )
    all_args = []
    for result in results:
        field = result[0]
        
        # DB merges these fields together in distinct, need to split them out
        src_fname, src_line, dst_fname, dst_line = field.split(",")
        dst_line = dst_line[:-1] #Remove trailing commas
        src_fname = src_fname[1:] #Remove leading (
        
        # Get CDFs
        cdf_vals1 = get_cdf( conn, src_fname, int(src_line), dst_fname, int(dst_line), run_id1 )
        cdf_vals2 = get_cdf( conn, src_fname, int(src_line), dst_fname, int(dst_line), run_id2 )
        if not cdf_vals1 or not cdf_vals2:
            continue

        cdf_vals1 = cdf_vals1[0][0]
        cdf_vals2 = cdf_vals2[0][0]

        all_args.append( (src_fname, src_line, dst_fname, dst_line, cdf_vals1, cdf_vals2, dist_mat, normalize) )
    emd_scores = proc_pool.map( do_emd, all_args )
    return emd_scores

### Variable Order Stuff.
class VariableOrderTransition:
    """Variable Order Transition (s-k,...,s) - > s'"""
    def __init__( self, prior_events: List[FileLocation], next_event: FileLocation ):
        self.prior_events = prior_events
        self.next_event = next_event
    def __repr__( self ) -> str:
        return "({})->{}".format( ",".join([ str(ev) for ev in self.prior_events ]), self.next_event )
    def __eq__( self, obj: Any ) -> bool:
        return isinstance( obj, VariableOrderTransition) and self.prior_events == obj.prior_events and self.next_event == obj.next_event
    def __ne__( self, obj: Any ) -> bool:
        return not self == obj
    def __hash__( self ) -> int:
        return hash( (tuple(self.prior_events), self.next_event) )


    def could_transition_to( self, other_transition: 'VariableOrderTransition' ) -> bool:
        """Given a different transition, see if this transition ends at the other transition's sequence start.
        For example, [1,2,3]->4's ending leads to [2,3,4]->X, or [3,4]->Y.
        This method is useful in determining which of our transition may lead to 'other_transition'
        """
        my_prior_seq = self.prior_events
        my_seq_len = len(my_prior_seq)
        other_prior_seq = other_transition.prior_events
        other_seq_len = len(other_prior_seq)

        # Do I transition to their last item
        if self.next_event != other_prior_seq[-1]:
            return False

        # Example:
        # Other_seq_len = 3.
        # for i in range(2)
        # 0,1
        for i in range(other_seq_len-1):
            # my_prior_seq[2-1-0], other_seq_len[3-2-0] => my_prior_seq[1], other_prior_seq[1]
            # my_prior_seq[2-1-1], other_prior_seq[3-2-1] = my_prior_seq[0], other_prior_seq[0]
            my_pos_to_check = my_seq_len-1-i
            their_pos_to_check = other_seq_len-2-i

            if my_pos_to_check < 0:
                # They still have more sequences to check, but we don't.
                # We can't guarantee that we end in their start.
                return False
                
            assert my_pos_to_check >= 0
            assert their_pos_to_check >= 0
            if my_prior_seq[my_pos_to_check] != other_prior_seq[their_pos_to_check]:
                return False

        return True
        
    def could_be_reduced_to( self, other_transition: 'VariableOrderTransition' ) -> bool:
        """Given a different transition, see if this transition can be reduced to the given transition
        For example, [1,2,3,4]->5 could be reduced to [2,3,4] -> 5 or [4] -> 5
        """
        if self.next_event != other_transition.next_event:
            return False
        
        my_prior_seq = self.prior_events
        my_seq_len = len(my_prior_seq)
        other_prior_seq = other_transition.prior_events
        other_seq_len = len(other_prior_seq)

        if other_seq_len > my_seq_len:
            return False

        for i in range(other_seq_len):
            if my_prior_seq[my_seq_len-1-i] != other_prior_seq[other_seq_len-1-i]:
                return False

        return True

class VariableOrderTransitionIndex:
    """An index from a tuple of prior events (s-k,...,s) to a list of transitions."""

    def __init__( self ) -> None:
        self.vot_prior_index = {} # type: Dict[Tuple[FileLocation, ...], List[VariableOrderTransition]]

    def add_transition( self, vo_transition: VariableOrderTransition ) -> None:
        key = tuple(vo_transition.prior_events)
        if not key in self.vot_prior_index:
            self.vot_prior_index[key] = []
        self.vot_prior_index[ key ].append( vo_transition )

    def find_all_transitions( self, prior_events: Tuple[FileLocation, ...],
            exact_matches_only: bool =False ) -> List[VariableOrderTransition]:
        key = prior_events[:] # copy
        while len(key) > 0:
            if key in self.vot_prior_index:
                return self.vot_prior_index[ key ]
            key = key[1:] # chop first element
            if exact_matches_only:
                break
        return []

    def is_in_index( self, event_sequence: Tuple[FileLocation, ...] ) -> bool:
        return event_sequence in self.vot_prior_index

def build_vot_prior_index( vo_transitions: Iterable[VariableOrderTransition] ) -> VariableOrderTransitionIndex:
    vot_index = VariableOrderTransitionIndex()
    for vo_transition in vo_transitions:
        vot_index.add_transition( vo_transition )
    return vot_index

def get_all_unique_vo_transitions( vo_transitions1 : Dict[VariableOrderTransition, int], vo_transitions2: Dict[VariableOrderTransition, int] ) -> Set[VariableOrderTransition]:
    unique_vo_transitions = set([]) # type: Set[VariableOrderTransition]
    matched_subtransition = set([]) # type: Set[VariableOrderTransition]
    transitions_without_matches = set([]) # type: Set[VariableOrderTransition]

    def get_all_unique_vo_transitions_sub(
            vo_transitions_i: Dict[VariableOrderTransition, int],
            vo_transitions_j: Dict[VariableOrderTransition, int],
            unique_vo_transitions: Set[VariableOrderTransition],
            matched_subtransition: Set[VariableOrderTransition],
            transitions_without_matches: Set[VariableOrderTransition] ) -> None:
        for vo_transition in vo_transitions_i:
            if vo_transition in vo_transitions_j:
                unique_vo_transitions.add( vo_transition )
            else:
                # This transition may not be present in vo_transitions_j because:
                # i) The transition never occurred in vo_transitions_j
                # ii) The transition occurred, but was reduced in vo_transitions_j but not in vo_transitions_i
                # iii) This is a reduced version of a transition that happened in vo_transitions_j
                # We want only the maximum length edition of this sequence.
                # To do so, compute subsets of this key and try to match them in vo_transitions_j.
                prior_events = vo_transition.prior_events[1:]
                while len(prior_events) > 0:
                    reduced_transition = VariableOrderTransition( prior_events, vo_transition.next_event )
                    if reduced_transition in vo_transitions_j:
                        # We matched on a reduced version, append the full version
                        unique_vo_transitions.add( vo_transition )
                        matched_subtransition.add( reduced_transition )
                        break
                    # Chop, go again
                    prior_events = prior_events[1:]

                # Didn't match on any subset. Either we are the subset, or this transition didn't happen in vo_transitions_j.
                # Need to handle second case where we should still add this transition (can check if someone matched us!)
                transitions_without_matches.add( vo_transition )

    get_all_unique_vo_transitions_sub( vo_transitions1, vo_transitions2, unique_vo_transitions, matched_subtransition, transitions_without_matches )
    get_all_unique_vo_transitions_sub( vo_transitions2, vo_transitions1, unique_vo_transitions, matched_subtransition, transitions_without_matches )

    for vo_transition in transitions_without_matches:
        if vo_transition not in matched_subtransition:
            unique_vo_transitions.add( vo_transition )

    return unique_vo_transitions

class VOTransitionDiffRecord:
    def __init__(
        self,
        transition: VariableOrderTransition,
        left_prob: float,
        right_prob: float,
        left_count_match: int,
        left_count_miss: int,
        right_count_match: int,
        right_count_miss: int,  
        left_metrics_datas: List[ Dict[str,MetricsFileData.MetricsData] ]=[],
        right_metrics_datas: List[ Dict[str,MetricsFileData.MetricsData] ]=[]
    ):
        self.transition = transition
        self.left_prob = left_prob
        self.right_prob = right_prob
        self.left_count = left_count_match
        self.right_count = right_count_match
        self.left_miss_count = left_count_miss
        self.right_miss_count = right_count_miss
        self.left_metrics_datas = left_metrics_datas
        self.right_metrics_datas = right_metrics_datas

        self.left_metrics_means = {} # type: Dict[str, float]
        self.right_metrics_means = {} # type: Dict[str, float]
        self.left_metrics_tot = {} # type: Dict[str, float]
        self.right_metrics_tot = {} # type: Dict[str, float]
        self.left_metrics_counts = {} # type: Dict[str, int]
        self.right_metrics_counts = {} # type: Dict[str, int]
        self.metrics_cdf_diffs = {} # type: Dict[str, float]

        self.score = 0.
        self.stat = 10000

        if left_prob == right_prob:
            self.score = 1.0
        elif left_prob > 0 and right_prob > 0:
            self.score = max( left_prob / right_prob, right_prob / left_prob )
        else:
            self.score = 5.0
        left_metric_names = set()
        for md in left_metrics_datas:
            for metric_name in md.keys():
                left_metric_names.add( metric_name )
        for metric_name in left_metric_names:
            self.left_metrics_counts[metric_name] = sum([md[metric_name].count for md in left_metrics_datas if metric_name in md])
            expectation_value = 0.
            if self.left_metrics_counts[metric_name] > 0:
                for md in left_metrics_datas:
                    if metric_name in md:
                        frac = float(md[metric_name].count) / self.left_metrics_counts[metric_name]
                        expectation_value += frac * np.mean(md[metric_name].sampled_vals)
            self.left_metrics_means[metric_name] = expectation_value
            self.left_metrics_tot[metric_name]=self.left_metrics_counts[metric_name] * expectation_value

        right_metric_names = set()
        for md in right_metrics_datas:
            for metric_name in md.keys():
                right_metric_names.add( metric_name )

        for metric_name in right_metric_names:
            self.right_metrics_counts[metric_name] = sum([md[metric_name].count for md in right_metrics_datas if metric_name in md])
            expectation_value = 0.
            if self.right_metrics_counts[metric_name] > 0:
                for md in right_metrics_datas:
                    if metric_name in md:
                        frac = float(md[metric_name].count) / self.right_metrics_counts[metric_name]
                        expectation_value += frac * np.mean(md[metric_name].sampled_vals)
            self.right_metrics_means[metric_name] = expectation_value
            self.right_metrics_tot[metric_name]=self.right_metrics_counts[metric_name] * expectation_value


def create_vot_diff_record(
        maximal_vo_transition: VariableOrderTransition,
        left_prob: float,
        right_prob: float,
        left_count_match: int,
        left_count_miss: int,
        right_count_match: int,
        right_count_miss: int,
        left_metrics_datas: List[ Dict[str,MetricsFileData.MetricsData ] ]=[],
        right_metrics_datas: List[ Dict[str, MetricsFileData.MetricsData] ]=[]
    ) -> VOTransitionDiffRecord:

    return VOTransitionDiffRecord( maximal_vo_transition, left_prob, right_prob, left_count_match, left_count_miss, right_count_match, right_count_miss, left_metrics_datas, right_metrics_datas )

def compute_vot_diff( vo_transitions1: Dict[VariableOrderTransition, int], vo_transitions2: Dict[VariableOrderTransition, int], vo_transitions_metrics1: Dict[VariableOrderTransition, Dict[ str, MetricsFileData.MetricsData]], vo_transitions_metrics2: Dict[VariableOrderTransition, Dict[ str, MetricsFileData.MetricsData]]) -> List[ VOTransitionDiffRecord ]:

    vot_diff_records = [] # type: List[ VOTransitionDiffRecord ]
    all_vo_transitions = get_all_unique_vo_transitions( vo_transitions1, vo_transitions2 )

    # A VariableOrderTransition consists of (s-k,...,s) -> s'.
    # We want to compare probabilities of making this particular transition (e.g. P(s'|s-k,...,s)), but we have only counts
    # Since P(s'|s-k,...s) = (s' from s-k,...s)/sum(anything from s-k,...,s), we need all VariableOrderTransition with the same prior events.
    # Let's build this index.
    vot_prior_event_index1 = build_vot_prior_index( vo_transitions1.keys() )
    vot_prior_event_index2 = build_vot_prior_index( vo_transitions2.keys() )

    for vo_transition in all_vo_transitions:
        # Suppose this transition is from (s-k,...,s)->s'. We need to obtain the 
        # the probability of this transition from both VariableOrderMarkovGraphs.
        # To do so, we need to find the count of moving from (s-k,...,s)->X for all
        # X, dividing that from the count of going from (s-k,...,s)->s'.

        # The problem is that either graph (or both) may not have (s-k,...,s) in it.
        # This could be because the transition's order has been reduced, or because the
        # the sequence (s-k,...,s) never occurred.

        # We know, however, that if (s-k,...,s)->X has been reduced to
        # (s-k-1,...,s)->X, it holds for all X. So, if we ever transitioned to s' from
        # (s-k,...,s)->s', then it would either show up at (s-k,...,s)->s' or as
        # (s-k-m,...,s)->s' for some m, and that the probability of (s-k-m,...,s)->s' is
        # the same as (s-k,...,s)->s'. So, search for (s-k,...s), (s-k-1,...,s), ... 
        # sequences until we find one, and use that to compute the transition probability to s',
        # which we know is the same.

        all_transitions_for_vom1 = vot_prior_event_index1.find_all_transitions( tuple(vo_transition.prior_events) )
        all_transitions_for_vom2 = vot_prior_event_index2.find_all_transitions( tuple(vo_transition.prior_events) )

        # We may have reduced the transition in one model, but not the other. Use the matching_transition we retrieved.
        tr_count_vom1 = 0
        this_tr_count_vom1 = 0
        have_match = False
        left_metrics_data = None
        for matching_transition in all_transitions_for_vom1:
            assert matching_transition in vo_transitions1
            assert matching_transition in vo_transitions_metrics1
            tr_count_vom1 += vo_transitions1[ matching_transition ]
            if matching_transition.next_event == vo_transition.next_event:
                this_tr_count_vom1 = vo_transitions1[ matching_transition ]
                left_metrics_data = vo_transitions_metrics1[ matching_transition ]
                have_match = True
        prob_tr_vom1 = 0.
        if len(all_transitions_for_vom1) > 0:
            prob_tr_vom1 = float( this_tr_count_vom1 ) / tr_count_vom1
            
        tr_count_vom2 = 0
        this_tr_count_vom2 = 0
        right_metrics_data = None
        for matching_transition in all_transitions_for_vom2:
            assert matching_transition in vo_transitions2
            assert matching_transition in vo_transitions_metrics2
            tr_count_vom2 += vo_transitions2[ matching_transition ]
            if matching_transition.next_event == vo_transition.next_event:
                this_tr_count_vom2 = vo_transitions2[ matching_transition ]
                right_metrics_data = vo_transitions_metrics2[ matching_transition ]
                have_match = True

        assert have_match

        prob_tr_vom2 = 0.
        if len(all_transitions_for_vom2) > 0:
            prob_tr_vom2 = float( this_tr_count_vom2 ) / tr_count_vom2

        left_metrics_datas = []
        right_metrics_datas = []
        if left_metrics_data:
            left_metrics_datas.append( left_metrics_data )
        if right_metrics_data:
            right_metrics_datas.append( right_metrics_data )
        # create vo_diff_record
        vot_diff_records.append(
            create_vot_diff_record( vo_transition, prob_tr_vom1, prob_tr_vom2, this_tr_count_vom1, tr_count_vom1-this_tr_count_vom1, this_tr_count_vom2, tr_count_vom2-this_tr_count_vom2, left_metrics_datas, right_metrics_datas ) )

    return vot_diff_records

class InvalidModelException(Exception):
    pass

def obtain_metric_samples_from_reservoir( metric_data: MetricsFileData.MetricsData, metric_tot_count: int ) -> List[float]:
    """When merging two metric reservoirs together for a particualr metric, obtain this many samples
    from the provided metric_data. Will scale proportionally for the number of sampled values in metric_data compared to the
    overall number of samples we are merging together in metric_tot_count."""

    assert metric_tot_count >= metric_data.count
    if metric_tot_count == metric_data.count:
        return metric_data.sampled_vals

    samples_to_obtain = min(math.floor((float(metric_data.count) / metric_tot_count) * 1000), metric_data.count)
    value_pool = [] # type: List[float]

    rand_indexes = np.random.randint(0,len(metric_data.sampled_vals), samples_to_obtain)
    for index in rand_indexes:
        value_pool.append( metric_data.sampled_vals[ index ] )
    return value_pool

def merge_metric_reservoirs( multiple_metrics_data_list: Iterable[Dict[str,MetricsFileData.MetricsData]] ) -> Dict[str,MetricsFileData.MetricsData]:
    """Given two dictionaries of MetricsData for named metrics, merge them together.
    If we exceed the maximum reservoir size, doing so will involve subsampling."""
    new_transition_metrics = {} # type: Dict[str, MetricsFileData.MetricsData]
    metric_name_counts = {} # type: Dict[str, int ]
    metric_names = set([]) # type: Set[str]
    metric_value_pools = {} # type:  Dict[str,List[float]]

    # Obtain all metric names and counts for these names
    for multiple_metric_data in multiple_metrics_data_list:
        metric_names.update( multiple_metric_data.keys() )
        for metric_name in multiple_metric_data.keys():
            if not metric_name in metric_name_counts:
                metric_name_counts[ metric_name ] = 0
                metric_value_pools[ metric_name ] = []
            metric_name_counts[ metric_name ] += multiple_metric_data[ metric_name ].count

    # Proportionally subsample or copy
    for metric_name in metric_names:
        for multiple_metric_data in multiple_metrics_data_list:
            if metric_name in multiple_metric_data.keys():
                this_md = multiple_metric_data[ metric_name ]
                if this_md.count > 0:
                    metric_value_pools[ metric_name ].extend( obtain_metric_samples_from_reservoir( this_md, metric_name_counts[ metric_name ] ) )

    # Convert metric_value_pools to md_dict
    for metric_name in metric_names:
        combined_md = MetricsFileData.MetricsData( metric_name_counts[ metric_name ], metric_value_pools[ metric_name ] )
        new_transition_metrics[ metric_name ] = combined_md

    return new_transition_metrics

class VariableOrderMarkovGraph:
    """A Markov Graph where the transitions between nodes may rely on a variable number of previous nodes.
    A traditional MarkovGraph uses P(s'|s), but this Markov Graph is P(s'|s,s-1,s-2...s-k) AND k is variable depending on s'"""

    def __init__( self, epsilon: float, sample_reduction_threshold: int, max_order: int):
        self.events = {} # type: Dict[FileLocation, EventRecord]
        self.transitions = {} # type: Dict[VariableOrderTransition, int]
        self.transitions_metrics = {} # type: Dict[VariableOrderTransition, Dict[ str, MetricsFileData.MetricsData]]
        self.epsilon = epsilon 
        self.sample_reduction_threshold = sample_reduction_threshold
        self.max_order = max_order
        self.model_expanded = False
        self.epoch = -1
        self.event_file_map = {} # type: Dict[int, FileLocation]
        self.event_level_map = {} # type: Dict[FileLocation, int]

    def set_epoch( self, epoch: int ) -> None:
        self.epoch = epoch

    def serialize( self ) -> bytes: 
        return pickle.dumps( self )

    @staticmethod
    def deserialize( data: bytes ) -> 'VariableOrderMarkovGraph':
        model = pickle.loads( data ) # type: VariableOrderMarkovGraph
        model.check_valid_model()
        return model

    @staticmethod
    def from_file( epsilon: float, sample_threshold: int, max_order: int, event_count_map: Dict[FileLocation, int], event_transition_map: Dict[VariableOrderTransition, int], event_file_map: Dict[int, FileLocation], event_level_map: Dict[FileLocation, int] ) -> 'VariableOrderMarkovGraph':
        vom = VariableOrderMarkovGraph( epsilon, sample_threshold, max_order )
        total_event_count = 0

        for event, count in event_count_map.items():
            total_event_count += count

        for event, count in event_count_map.items():
            rec = EventRecord( event.fname, event.line_number, count, float(count) / total_event_count )
            vom.events[ event ] = rec

        vom.transitions = event_transition_map
        vom.event_file_map = event_file_map
        vom.event_level_map = event_level_map
        return vom

    def merge_in_events( self, other_vom: 'VariableOrderMarkovGraph' ) -> None:

        assert self.max_order == other_vom.max_order
        
        # Start by updating the counts, we'll loop around later to adjust probabilities
        for event, record in other_vom.events.items():
            if event in self.events:
                self.events[ event ].count += record.count
            else:
                self.events[ event ] = EventRecord( event.fname, event.line_number, record.count, 0. )

        # Event counts are all updated, get total event count.
        total_event_count = 0
        for event, record in self.events.items():
            total_event_count += record.count

        # update probabilities
        for event, record in self.events.items():
            record.prob = float(record.count) / total_event_count

        self.event_level_map.update( other_vom.event_level_map )

    def find_transitions_by_next_event( self, next_event: int ) -> List[VariableOrderTransition]:
        return [ transition for transition in self.transitions if transition.next_event == next_event ]

    def have_reduced_form_of_transition( self, transition: VariableOrderTransition ) -> Tuple[bool, Optional[VariableOrderTransition]]:
        prior_event_seq = transition.prior_events
        while len(prior_event_seq) > 1:
            prior_event_seq = prior_event_seq[1:]
            tmp_transition = VariableOrderTransition( prior_event_seq, transition.next_event )
            if tmp_transition in self.transitions:
                return (True, tmp_transition)
        return (False, None)

    def have_expanded_form_of_transition( self, transition: VariableOrderTransition ) -> Tuple[bool, List[VariableOrderTransition]]:
        # We could build an index and try to match, but this hopefully happens infrequently and takes place offline.
        
        other_event_seq = transition.prior_events
        matches = [] # type: List[VariableOrderTransition]
        for loc_transition in self.transitions:
            loc_event_seq = loc_transition.prior_events
            if len(loc_event_seq) <= len(other_event_seq):
                continue
            is_match = True
            # The end needs to match.
            for i in range(len(other_event_seq)):
                if other_event_seq[len(other_event_seq)-i-1] != loc_event_seq[len(loc_event_seq)-i-1]:
                    is_match = False
                    break
            if is_match:
                matches.append( loc_transition )
        return (len(matches) > 0, matches)

    def merge_in_transitions( self, other_vom: 'VariableOrderMarkovGraph' ) -> None:
        """Merge the transitions (and associated metrics) from other_vom into this VariableOrderMarkovGraph"""

        log.debug( "%s merging in %s", self.transitions, other_vom.transitions )
        self.expand_model()
        other_vom.expand_model()

        for transition in other_vom.transitions:
            if transition in self.transitions:
                # We both have this transition. Time to merge the reservoirs.
                self.transitions[ transition ] += other_vom.transitions[ transition ]
                new_transition_metrics = merge_metric_reservoirs( [self.transitions_metrics[ transition ], other_vom.transitions_metrics[ transition ] ] )
                self.transitions_metrics[ transition ] = new_transition_metrics
            else:
                # We don't have this transition, copy it in
                self.transitions[ transition ] = other_vom.transitions[ transition ]
                self.transitions_metrics[ transition ] = other_vom.transitions_metrics[ transition ]
        # We might have transitions they don't. We don't need to do anything about that.

    def expand_model_by_one_order( self ) -> bool:

        if self.model_expanded:
            return False

        min_order = min( [ len( tr.prior_events ) for tr in self.transitions ], default=self.max_order )

        if min_order == self.max_order:
            self.model_expanded = True
            return False
        next_order = min_order + 1

        new_transitions = {} # type: Dict[VariableOrderTransition,int]
        new_transitions_metrics = {} # type: Dict[VariableOrderTransition, Dict[str,MetricsFileData.MetricsData]]
        transitions_to_skip = set([]) # type: Set[VariableOrderTransition]
        prior_index = build_vot_prior_index( self.transitions )

        for transition in self.transitions:
            assert len(transition.prior_events) <= self.max_order

            # When we expand a transition, we expand a bunch at a time.
            # If we've already handled this transtiion, skip it.
            if transition in transitions_to_skip:
                continue

            if len(transition.prior_events) >= next_order:
                # No need to expand, copy it over.
                new_transitions[ transition ] = self.transitions[ transition ]
                new_transitions_metrics[ transition ] = self.transitions_metrics[ transition ]
                continue

            # OK, time to expand.

            # Find everything this reduced transition transitions to. We are going to expand all these transitions back up.
            all_transitions_that_share_my_prior_seq = prior_index.find_all_transitions( tuple(transition.prior_events), exact_matches_only=True ) # exact matches only, do not recurse on prior_events
            next_event_list = [tr.next_event for tr in all_transitions_that_share_my_prior_seq]
            for event in next_event_list:
                # Sanity check
                tr = VariableOrderTransition( transition.prior_events[:], event )
                assert tr in self.transitions
            log.debug( "Found %d events that prior event sequence %s transitions to: %s", len(next_event_list), str(tuple(transition.prior_events)), str(next_event_list) )
    
            # What do we know?
            # I reduced from (s_k,...,s) to (s_k-m,...,s) only if the transition probability is independent of whatever (s_k,...,s_k-m+1) events
            # happened beforehand. While I might think that I could stick *any* (s_k,...,s_k-m+1) there because P(s''|s_k...s)=P(s''|s_k-m...s),
            # picking (s_k,...,s_k-m+1) at random isn't great because the overall sequence of events (s_k,...,s) may never have occurred in practice.
            # However, we know what sequences of (s_k...,s_k-m) are possible because they would be encoded in other transitions. So, what transitions to
            # s_k-m

            # Find any unique transition that ends in prior_events[0]...prior_events[len-2] and transitions to prior_events[len-1]
            transitions_that_end_in_my_start = [ tr for tr in self.transitions if tr.could_transition_to( transition ) ]
            log.debug( "Found transitions that end in my start: %s, for %s", transitions_that_end_in_my_start, transition )
            unique_priors = set([])
            for tr in transitions_that_end_in_my_start:
                # Our sequence will be shifted to the right one, because whatever transitions
                # to me will have the last event of my sequence as their next_event.
                trunc_seq_len = len(transition.prior_events)-1

                # walk back the length of our sequence to find what came there before.
                the_prior_event = tr.prior_events[len(tr.prior_events)-1-trunc_seq_len]
                unique_priors.add( the_prior_event )
            log.debug( "Found %d other transitions that could lead to reduced transition %s: %s", len(transitions_that_end_in_my_start), transition, transitions_that_end_in_my_start )
            log.debug( "Of those, there are %d unique prior events: %s", len(unique_priors), str(unique_priors) )

            # There's some subtleties here.
            # transition is instantiated with a particular version of next_event, not necessarily *this* next_event
            # While it will have the same prior event list as the expanded version we want to add, the count for *this* transition 
            # is not the same as the count for the transition with *this* next_event.
            for next_event in next_event_list:
                log.debug("Considering next_event: %s", next_event )
                transitions_to_remove = set([]) #type: Set[VariableOrderTransition]

                # Construct what all of the expanded transitions are and add them
                for unique_prior in unique_priors:
                    prior_event_seq = transition.prior_events[:] # copy
                    expanded_event_seq = prior_event_seq[:] # copy
                    expanded_event_seq.insert(0,unique_prior)
                    log.debug( "Using unique prior: %s", unique_prior )
                    expanded_transition = VariableOrderTransition( expanded_event_seq, next_event )
                    log.debug( "Constructed expanded event transition %s", expanded_transition )

                    # We need to determine the number of transitions for this expanded transition.
                    # Concretely, we are in the position where we have: [E1,E2]->E3 and we want to expand it
                    # to [E0,E1,E2]->E3. To do this, we need to find the count of times that [E0,E1]->E2 because
                    # that gives us [E0,E1,E2], and multiply that by P(E3|E1,E2) because we know it is equal to
                    # P(E3|E0,E1,E2).

                    # We already have transitions_that_end_in_my_start.

                    # Suppose we are in the case where we have: [E1,E2]->E3 that we are trying to expand and
                    # we know that [E0',E0,E1]->E2. How do we figure out [E0,E1,E2]->E3?
                    # count(E0',E0,E1->E2) = count(E0',E0,E1,E2)
                    # sum over all E0' gives us the count.

                    # [unique_prior,... seq without end el... ] -> end el

                    log.debug( "Walking over all transitions that end in my start: %s", transitions_that_end_in_my_start )
                    log.debug( "Recall that original transition is %s", transition )

                    # After we do the expansion, only some of the things that transition to us will match the newly expanded transition.
                    # Figure out what those are.
                    trs_that_match = []
                    for tr in transitions_that_end_in_my_start:
                        match = True
                        for i in range(len(expanded_transition.prior_events)-1):
                            if tr.prior_events[-1-i] != expanded_transition.prior_events[-i-2]:
                                match = False
                                break
                        if match:
                            trs_that_match.append( tr )

                    log.debug( "Determined the following trs match my expanded transition: %s", trs_that_match )

                    # This is the total number of times we've transitioned to the current prior event sequence.
                    # count(E0,E1,E2) above.
                    tot_tr_count_for_this_prior = sum( [ self.transitions[tr] for tr in trs_that_match ] )
                    log.debug( "Number of times these trs have occurred %d", tot_tr_count_for_this_prior )
                    
                    log.debug( "Now we to generate the right transition count to get the same probability as the reduced transition.")
                    # compute probability
                    next_event_tr_count = 0.
                    tot_tr_count_for_this_seq = 0.
                    log.debug( "Iterating over all that original transition's sequence: %s", all_transitions_that_share_my_prior_seq )
                    log.debug( "Looking to match subsequent event: %s", next_event )
                    for tr in all_transitions_that_share_my_prior_seq:
                        log.debug( "Considering tr %s", tr)
                        if tr.next_event == next_event:
                            log.debug( "Matched %s, added count %d", tr, self.transitions[tr] )
                            next_event_tr_count += self.transitions[tr]
                        log.debug( "Adding %s count to tot_tr_count_for_this_seq: %d", tr, self.transitions[tr] )
                        tot_tr_count_for_this_seq += self.transitions[tr]

                    log.debug( "Tot tr count for this seq: %d", tot_tr_count_for_this_seq )
                    tr_prob = float(next_event_tr_count)/tot_tr_count_for_this_seq
                    true_count_for_expanded_seq = math.floor(tr_prob * tot_tr_count_for_this_prior)
                    if true_count_for_expanded_seq == 0:
                        log.debug( "Expanded Seq %s got count 0!", expanded_transition )
                        log.debug( "Next Event Tr Count %d tot_tr_count_for_this_seq %d", next_event_tr_count, tot_tr_count_for_this_seq )
                        log.debug( "TR_PROB: %f", tr_prob )
                        log.debug( "TR_COUNT_FOR_PRIOR %d", tot_tr_count_for_this_prior )
                        log.debug( "THIS IS NOT A REAL TRANSITION, skipping" )
                        continue

                    new_transitions[ expanded_transition ] = true_count_for_expanded_seq

                    # The true count on the expanded transition might be lower than the number of 
                    # metrics we have recorded for a reduced transition (if it was combined).
                    # There's no way for us to attribute which data points came from which
                    # expanded transition, but we assume that if we reduced its because the 
                    # distribution of the metrics matched.

                    # We could subsample the metrics to obtain a reservoir of the right size
                    # if the true count <= max_reservoir size, but all that would do is hurt the
                    # quality of the distribution we've recorded. Provided we subsample with the
                    # true count if we merge this vom with another later, all should be OK.

                    if next_event == transition.next_event:
                        log.debug( "Next event is a match!" )
                        new_transitions_metrics[ expanded_transition ] = copy.deepcopy( self.transitions_metrics[ transition ] )
                        for metric_name in new_transitions_metrics[ expanded_transition ].keys():
                            cur_metric_count = self.transitions_metrics[ transition ][ metric_name ].count
                            proportion_of_matches = float( self.transitions_metrics[ transition ][ metric_name ].count ) / self.transitions[ transition ]
                            new_transitions_metrics[ expanded_transition ][ metric_name ].count = math.floor( proportion_of_matches * true_count_for_expanded_seq )
                        log.debug( "Copied over event! %s", expanded_transition )
                        transitions_to_skip.add( transition )
                    else:
                        log.debug( "Next event is NOT a match!" )
                        reduced_transition_for_next_event = VariableOrderTransition( prior_event_seq, next_event )
                        assert reduced_transition_for_next_event in self.transitions
                        new_transitions_metrics[ expanded_transition ] = copy.deepcopy( self.transitions_metrics[ reduced_transition_for_next_event ] )
                        for metric_name in new_transitions_metrics[ expanded_transition ].keys():
                            cur_metric_count = self.transitions_metrics[ reduced_transition_for_next_event ][ metric_name ].count
                            proportion_of_matches = float( self.transitions_metrics[ reduced_transition_for_next_event ][ metric_name ].count ) / self.transitions[ reduced_transition_for_next_event ]
                            new_transitions_metrics[ expanded_transition ][ metric_name ].count = math.floor( proportion_of_matches * true_count_for_expanded_seq )

                        transitions_to_skip.add( reduced_transition_for_next_event )

        self.transitions = new_transitions
        self.transitions_metrics = new_transitions_metrics
        assert len(self.transitions.keys()) == len(self.transitions_metrics.keys())

        # Is there more expansion to be done?
        return next_order < self.max_order


    def expand_model( self ) -> None:
        """Expand every reduced transition back up to max order."""
        while self.expand_model_by_one_order():
            pass
        self.model_expanded = True

    def reduce_model( self ) -> None:
        if not self.model_expanded:
            return

        self.reduce_model_sub()
        self.reduce_model_sub()

        self.model_expanded = False


    def reduce_model_sub( self ) -> bool:
        """Reduce the model down as much as possible while preserving probabilities."""

        new_transitions = {} # type: Dict[VariableOrderTransition, int]
        new_transitions_metrics_pool = {} # type: Dict[ VariableOrderTransition, List[Dict[str, MetricsFileData.MetricsData]]]
        new_transitions_metrics = {} # type: Dict[ VariableOrderTransition, Dict[str, MetricsFileData.MetricsData]]
        transitions_to_skip = set([]) # type: Set[VariableOrderTransition]

        did_reduce = False

        for transition in self.transitions:
            prior_event_seq = transition.prior_events
            vot_prior_event_index = build_vot_prior_index( self.transitions )
            all_transitions_that_share_my_prior_seq = vot_prior_event_index.find_all_transitions( tuple(prior_event_seq), exact_matches_only=True )

            total_count = 0.
            for tr in all_transitions_that_share_my_prior_seq:
                assert tr in self.transitions
                total_count += self.transitions[ tr ]

            assert transition in self.transitions
            cur_count = float(self.transitions[ transition ])

            tr_prob = cur_count / total_count

            all_unique_endings = set([ tr.next_event for tr in all_transitions_that_share_my_prior_seq ])

            # Now I need to find anything that ends in my sequence (except for the first event).
            all_matching_subtransitions = []
            seq_len = len(transition.prior_events)
            for tr in self.transitions:
                if tr.prior_events[-seq_len + 1:] == transition.prior_events[1:]:
                    all_matching_subtransitions.append( tr )

            can_reduce = True
            for ending_event in all_unique_endings:
            
                total_sub_count = 0.
                total_sub_tr_count = 0.
                for tr in all_matching_subtransitions:
                    total_sub_count += self.transitions[tr]
                    if tr.next_event == ending_event:
                        total_sub_tr_count += self.transitions[tr]

                sub_tr_prob = total_sub_tr_count / total_sub_count

                # FIXME: Should be window and sample constrained, instead of a strict
                # equality
                # This is just a command_client library, not the injection shim...
                # the injection shim performs model reductions while the system is running
                # this is just a helper library to do some offline tests so it doesn't matter
                if tr_prob != sub_tr_prob:
                    can_reduce = False
                    break
            if can_reduce:
                # Perform the reduction
                log.debug( "Can reduce %s", transition )
                reduced_transitions = {} # type: Dict[VariableOrderTransition, int]

                for tr in all_matching_subtransitions:
                    reduced_tr = VariableOrderTransition( tr.prior_events[1:], tr.next_event )
                    if reduced_tr in reduced_transitions:
                        reduced_transitions[ reduced_tr ] += self.transitions[ tr ]
                        new_transitions_metrics_pool[ reduced_tr ].append( self.transitions_metrics[tr] )
                    else:
                        reduced_transitions[ reduced_tr ] = self.transitions[ tr ]
                        new_transitions_metrics_pool[ reduced_tr ] = []

                        assert tr in self.transitions
                        assert tr in self.transitions_metrics

                        new_transitions_metrics_pool[ reduced_tr ].append( self.transitions_metrics[tr] )

                # We should skip any of the old transitions that we reduced.
                transitions_to_skip.update( reduced_transitions.keys() )
                new_transitions.update( reduced_transitions )                
                did_reduce = True
                    
            else:
                log.debug( "Cannot reduce %s", transition )
                new_transitions[ transition ] = self.transitions[ transition ]
                new_transitions_metrics[ transition ] = self.transitions_metrics[ transition ]

        self.transitions = new_transitions

        # Time to subsample the metrics we have
        for reduced_tr in new_transitions_metrics_pool.keys():
            new_transitions_metrics[ reduced_tr ] = {}
            new_transitions_metrics[ reduced_tr ] = merge_metric_reservoirs( new_transitions_metrics_pool[ reduced_tr ] )
        self.transitions_metrics = new_transitions_metrics

        self.model_expanded = False

        return did_reduce

    def merge_in_event_file_map( self, other_event_file_map: Dict[int, FileLocation] ) -> None:
        for event_id in other_event_file_map:
            if event_id not in self.event_file_map:
                self.event_file_map[ event_id ] = other_event_file_map[ event_id ]

    def merge_in( self, other_vom: 'VariableOrderMarkovGraph' ) -> None:
        self.merge_in_events( other_vom )
        self.merge_in_transitions( other_vom )
        self.merge_in_event_file_map( other_vom.event_file_map )

    def incorporate_transition_metrics( self, metrics_for_transitions: Dict[ MetricsFileData.MetricsTransitionKey, Dict[ str, MetricsFileData.MetricsData]] ) -> None:
        for transition_key in metrics_for_transitions.keys():
            metrics_data = metrics_for_transitions[ transition_key ]
            file_loc_seq = [ self.event_file_map[ event_id ] for event_id in transition_key.prior_event_seq ]
            dst_loc = self.event_file_map[ transition_key.next_event ]
            tr = VariableOrderTransition( file_loc_seq, dst_loc )
            assert tr in self.transitions
            assert not tr in self.transitions_metrics
            self.transitions_metrics[ tr ] = metrics_data

    def diff(
        self,
        other_vom: 'VariableOrderMarkovGraph',
        missing_events_level_score_mapper: Callable[[FileLocation, int, int, Dict[FileLocation, EventRecord], Dict[FileLocation, EventRecord]],float] = lambda loc, level, count, events1, events2: 5.0,
        log_level_rescaling_function: Callable[[FileLocation, int, float, Dict[FileLocation, EventRecord], Dict[FileLocation,EventRecord]],float] = lambda loc, level, score, events1, events2: score
    ) -> Tuple[List[EventDifferenceRecord],List[VOTransitionDiffRecord]]:

        if not self.model_expanded:
            self.expand_model()
        if not other_vom.model_expanded:
            other_vom.expand_model()

        combined_level_map = {} # type: Dict[FileLocation, int]
        try:
            combined_level_map = copy.deepcopy( self.event_level_map )
        except:
            pass
        
        try:
            combined_level_map.update( other_vom.event_level_map )
        except:
            pass

        event_diff_records = compute_event_difference(
            self.events,
            other_vom.events,
            combined_level_map,
            missing_events_level_score_mapper,
            log_level_rescaling_function
        )

        vot_diff_records = compute_vot_diff( self.transitions, other_vom.transitions, self.transitions_metrics, other_vom.transitions_metrics )
        return event_diff_records, vot_diff_records

    def is_similar_to(
        self,
        other_vom: 'VariableOrderMarkovGraph',
        threshold: float,
        missing_events_level_score_mapper: Callable[[FileLocation, int, int, Dict[FileLocation, EventRecord], Dict[FileLocation, EventRecord]],float] = lambda loc, level, count, events1, events2: 5.0,
        log_level_rescaling_function: Callable[[FileLocation, int, float, Dict[FileLocation, EventRecord], Dict[FileLocation, EventRecord]],float] = lambda loc, level, score, events1, events2: score
    ) -> bool:
        if not self.model_expanded:
            self.expand_model()
        if not other_vom.model_expanded:
            other_vom.expand_model()

        combined_level_map = {} # type: Dict[FileLocation, int]
        try:
            combined_level_map = copy.deepcopy( self.event_level_map )
        except:
            pass
        
        try:
            combined_level_map.update( other_vom.event_level_map )
        except:
            pass

        event_diff_records = compute_event_difference(
            self.events,
            other_vom.events,
            combined_level_map,
            missing_events_level_score_mapper,
            log_level_rescaling_function
        )
        is_similar = np.mean(np.array( [ event_record.prob_diff_val for event_record in event_diff_records] )) <= threshold # type: bool
        return is_similar

    def check_valid_model( self ) -> None:
        """Confirm that this variable order markov model is valid. That is,
        if it contains a transitions from (s_k,...,s), it does not contain transitions from
        (s_k-m,...s) for all m."""

        prior_vot_index = build_vot_prior_index( self.transitions )

        for transition in self.transitions.keys():
            prior_event_seq = transition.prior_events
            while len(prior_event_seq) > 1:
                prior_event_subseq = prior_event_seq[1:]
                if prior_vot_index.is_in_index( tuple(prior_event_subseq) ):
                    short_tr = prior_vot_index.find_all_transitions( tuple(prior_event_subseq) )[0]
                    raise InvalidModelException( "{} in model, but so is {}".format( short_tr, transition ) )
                prior_event_seq = prior_event_subseq


def process_event_line( line: str, file_map: Dict[int, FileLocation], count_map: Dict[FileLocation,int], level_map: Dict[FileLocation, int] ) -> None:
    """Process a line like "pg.c:1 = 1, 10" and add the filelocation and count to the right maps"""
    left, right = line.split("=")
    fname, line_number = left.split(":")
    loc = FileLocation( fname, int( line_number.strip() ) )
    identifier_str, count_str, level_str = right.lstrip().split(",")
    identifier = int( identifier_str )
    count = int( count_str )
    level = int( level_str.strip() )
    assert identifier not in file_map
    file_map[identifier] = loc
    count_map[loc] = count
    level_map[loc] = level

def process_transition_line( line: str, file_map: Dict[int,FileLocation], transition_map: Dict[VariableOrderTransition, int] ) -> None:
    """Process a line like "(1,2)->1: 2" and add the transition and count to the right maps"""
    left, right = line.split("->")

    # Skip over old transitions that we outputted.
    if not "(" in line:
        return

    prior_event_id_str = left.strip()
    prior_event_id_str = prior_event_id_str[1:-1] # chop off brackets
    prior_event_ids = [ int(prior_event_id) for prior_event_id in prior_event_id_str.split(",") ]

    transition, count_str = right.lstrip().split(":")
    transition_id = int( transition.lstrip() )
    count = int( count_str.lstrip() )

    prior_event_locs = [ file_map[ prior_event_id ] for prior_event_id in prior_event_ids ]
    right_loc = file_map[ transition_id ]

    vo_transition = VariableOrderTransition( prior_event_locs, right_loc )
    transition_map[vo_transition] = count

def process_dump_lines( lines: List[str] ) -> VariableOrderMarkovGraph:
    """ Read all the lines in a file and convert them into a FileEventSummary"""
    i = 0
    event_file_map = {} # type: Dict[int, FileLocation]
    event_count_map = {} # type: Dict[FileLocation, int]
    event_level_map = {} # type: Dict[FileLocation, int]
    event_transition_map = {} # type: Dict[VariableOrderTransition, int]

    while i < len(lines):
        # This is a transition line, break into transition processing
        if not "=" in lines[i]:
            break
        # This is still an event count line
        process_event_line( lines[i], event_file_map, event_count_map, event_level_map )
        i += 1

    epsilon, sample_threshold, max_order = lines[i].split()
    i += 1 

    while i < len(lines):
        process_transition_line( lines[i], event_file_map, event_transition_map )
        i += 1

    vom = VariableOrderMarkovGraph.from_file( float(epsilon), int(sample_threshold), int(max_order), event_count_map, event_transition_map, event_file_map, event_level_map )
    return vom

def read_single_im_dump( filename: str ) -> VariableOrderMarkovGraph:
    with open( filename, "r" ) as f:
        f_lines = f.readlines()
        vom = process_dump_lines( f_lines )
        return vom

def read_all_im_dumps( im_dir: str ) -> VariableOrderMarkovGraph:
    vom = None
    print( "Going to read_all_im_dumps." )
    for fname in glob.iglob( "{}/*.im.out*".format( im_dir ) ):
        print( fname )
        next_vom = read_single_im_dump( fname )
        next_vom.check_valid_model()
        if vom:
            vom.merge_in( next_vom )
            vom.check_valid_model()
        else:
            vom = next_vom
    assert vom is not None
    return vom
