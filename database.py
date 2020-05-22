"""
EDITS
Author: Jordan Ott 
Date: 5/20/2020
Description:
    Remove dependency of MongoDB
    Read and write to files for trial parameters and results


SHERPA is a Python library for hyperparameter tuning of machine learning models.
Copyright (C) 2018  Lars Hertel, Peter Sadowski, and Julian Collado.

This file is part of SHERPA.

SHERPA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SHERPA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SHERPA.  If not, see <http://www.gnu.org/licenses/>.
"""
import os 
import json 
from filelock import FileLock

import logging
import numpy
import subprocess
import time
import os
import socket
import warnings

try:
    from subprocess import DEVNULL # python 3
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')
import sherpa


dblogger = logging.getLogger(__name__)


class _Database(object):
    """
    Manages a Mongo-DB for storing metrics and delivering parameters to trials.

    The Mongo-DB contains one database that serves as a queue of future trials
    and one to store results of active and finished trials.

    Attributes:
        dbpath (str): the path where Mongo-DB stores its files.
        port (int): the port on which the Mongo-DB should run.
        reinstantiated (bool): whether an instance of the MongoDB is being loaded.
        mongodb_args (dict): keyword arguments to MongoDB
    """
    def __init__(self, db_dir, port=27010, reinstantiated=False,
                 mongodb_args={}):
        self.collected_results = set()
    
        self.dir = db_dir
        self.port = port
        self.reinstantiated = reinstantiated
        self.mongodb_args = mongodb_args
        
        self.trials = {}
        self.trials_file_name = os.path.join(self.dir, 'trials.json')
        self.results_file_name = os.path.join(self.dir, 'results.json')
            
        logging.disable(logging.CRITICAL)
        
        with FileLock(self.results_file_name + ".lock"):
            with open(self.results_file_name, 'w') as f:
                f.write('[\n')
            
        logging.disable(logging.NOTSET)
            
    def close(self):
        print('Closing MongoDB!')

    def start(self):
        """
        Runs the DB in a sub-process.
        """

        args = {"--" + k: v for k, v in self.mongodb_args.items()}
        if "--dbpath" in args:
            warnings.warn("Writing MongoDB to custom path {} instead of "
                          "output dir {}".format(args["--dbpath"], self.dir),
                          UserWarning)
        else:
            args["--dbpath"] = self.dir
        if "--logpath" in args:
            warnings.warn("Writing MongoDB logs to custom path {} instead of "
                          "output dir {}".format(
                args["--logpath"], os.path.join(self.dir, "log.txt")),
                UserWarning)
        else:
            args["--logpath"] = os.path.join(self.dir, "log.txt")
        if "--port" in args:
            warnings.warn("Starting MongoDB on port {} instead of "
                          "port {}. Set port via the db_port argument in "
                          "sherpa.optimize".format(args["--port"], self.port),
                          UserWarning)
        else:
            args["--port"] = str(self.port)

        dblogger.debug("Starting MongoDB...\nDIR:\t{}\nADDRESS:\t{}:{}".format(
            self.dir, socket.gethostname(), self.port))
        cmd = ['mongod']
        cmd += [str(item) for keyvalue in args.items() for item in keyvalue if item is not '']
        
        dblogger.debug("Starting MongoDB using command:{}".format(str(cmd)))

        time.sleep(1)
        self.check_db_status()
        if self.reinstantiated:
            self.get_new_results()

    def check_db_status(self):
        """
        Checks whether database is still running.
        """

    def get_new_results(self):
        """
        Checks database for new results.

        Returns:
            (list[dict]) where each dict is one row from the DB.
        """
        
        new_results = []
        if os.path.exists(self.results_file_name):
            
            logging.disable(logging.CRITICAL)
            
            with FileLock(self.results_file_name + '.lock'):
                with open(self.results_file_name, 'r') as f:
                    contents = f.read().strip()

                    if contents[-1] == '[':
                        tmp = '[]'
                    else:
                        tmp = contents[:-1] + '\n]'

                    results = json.loads(tmp)

                    for i, result in enumerate(results):
                        if i not in self.collected_results:
                            new_results.append(result)
                            self.collected_results.add(i)

            logging.disable(logging.NOTSET)
        
        return new_results

    def enqueue_trial(self, trial):
        """
        Puts a new trial in the queue for trial scripts to get.
        """
        trial = {'trial_id': trial.id,
                 'parameters': trial.parameters}
        self.trials[int(trial['trial_id'])] = trial['parameters']
        
        logging.disable(logging.CRITICAL)
        
        with FileLock(self.trials_file_name + ".lock"):
            with open(self.trials_file_name, 'w') as f:
                f.write(json.dumps(self.trials))
                
        logging.disable(logging.NOTSET)
        

    def add_for_stopping(self, trial_id):
        """
        Adds a trial for stopping.

        In the trial-script this will raise an exception causing the trial to
        stop.

        Args:
            trial_id (int): the ID of the trial to stop.
        """

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Client(object):
    """
    Registers a session with a Sherpa Study via the port of the database.

    This function is called from trial-scripts only.

    Attributes:
        host (str): the host that runs the database. Passed host, host set via
            environment variable or 'localhost' in that order.
        port (int): port that database is running on. Passed port, port set via
            environment variable or 27010 in that order.
    """
    def __init__(self, host=None, port=None, test_mode=False, **mongo_client_args):
        """
        Args:
            host (str): the host that runs the database. Generally not needed since
                the scheduler passes the DB-host as an environment variable.
            port (int): port that database is running on. Generally not needed since
                the scheduler passes the DB-port as an environment variable.
            test_mode (bool): mock the client, that is, get_trial returns a trial
                that is empty, keras_send_metrics accepts calls but does not do any-
                thing, as does send_metrics. Useful for trial script debugging.
        """
        self.test_mode = test_mode
        if not self.test_mode:
            host = host or os.environ.get('SHERPA_DB_HOST') or 'localhost'
            port = port or os.environ.get('SHERPA_DB_PORT') or 27010
            output_dir = os.environ.get('SHERPA_OUTPUT_DIR')
            
            self.trials_file_name = os.path.join(output_dir, 'trials.json')
            self.results_file_name = os.path.join(output_dir, 'results.json')

    def get_trial(self):
        """
        Returns the next trial from a Sherpa Study.

        Returns:
            sherpa.core.Trial: The trial to run.
        """
        if self.test_mode:
            return sherpa.Trial(id=1, parameters={})
        
        assert os.environ.get('SHERPA_TRIAL_ID'), "Environment-variable SHERPA_TRIAL_ID not found. Scheduler needs to set this variable in the environment when submitting a job"
        trial_id = int(os.environ.get('SHERPA_TRIAL_ID'))
        
        logging.disable(logging.CRITICAL)
        
        with FileLock(self.trials_file_name + '.lock'):
            with open(self.trials_file_name, 'r') as f:
                trials = json.load(f)
                return sherpa.Trial(id=trial_id, parameters=trials[str(trial_id)])
            
        logging.disable(logging.NOTSET)
            

    def send_metrics(self, trial, iteration, objective, context={}):
        """
        Sends metrics for a trial to database.

        Args:
            trial (sherpa.core.Trial): trial to send metrics for.
            iteration (int): the iteration e.g. epoch the metrics are for.
            objective (float): the objective value.
            context (dict): other metric-values.
        """
        if self.test_mode:
            return
        
        result = {'parameters': trial.parameters,
                  'trial_id': trial.id,
                  'objective': objective,
                  'iteration': iteration,
                  'context': context}
        # Convert float32 to float64.
        # Note: Keras ReduceLROnPlateau callback requires this.
        for k,v in context.items():
            if type(v) == numpy.float32:
                context[k] = numpy.float64(v)
        
        logging.disable(logging.CRITICAL)

        with FileLock(self.results_file_name + ".lock"):
            with open(self.results_file_name, 'a') as f:
                f.write(json.dumps(result))
                f.write(',\n')      
                
        logging.disable(logging.NOTSET)
        

    def keras_send_metrics(self, trial, objective_name, context_names=[]):
        """
        Keras Callbacks to send metrics to SHERPA.

        Args:
            trial (sherpa.core.Trial): trial to send metrics for.
            objective_name (str): the name of the objective e.g. ``loss``,
                ``val_loss``, or any of the submitted metrics.
            context_names (list[str]): names of all other metrics to be
                monitored.
        """        
        import keras.callbacks
        send_call = lambda epoch, logs: self.send_metrics(trial=trial,
                                                          iteration=epoch,
                                                          objective=logs[objective_name],
                                                          context={n: logs[n] for n in context_names})
        return keras.callbacks.LambdaCallback(on_epoch_end=send_call)
