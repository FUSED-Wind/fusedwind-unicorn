
import os

try:
    from mpi4py import MPI
except:
    print('It seems that we are not able to import MPI')
    MPI = None

def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Define MPI message tags
tags = enum('READY', 'DONE', 'EXIT', 'START')

# This is a case iterator that will run under MPI

# This is a generic case runner
class FUSED_MPI_Cases(object):

    def __init__(self, jobs=[], comm=None, prePostExec=None):
        super(FUSED_MPI_Cases, self).__init__()

        self.comm=comm
        if not MPI is None and self.comm==None:
            self.comm=MPI.COMM_WORLD

        # set the data
        self.jobs = jobs
        self.i_am_executing = False
        self.prePostExec = prePostExec

    def add_job(self, job):

        self.jobs.append(job)

    def get_job_count(self):

        return len(self.jobs)

    def get_job(self, at_id):

        return self.jobs[at_id]

    def pre_run(self):
        pass

    def post_run(self, job_list):
        # Note job list stores the rank that completed a given job
        pass

    def execute_job(self, job_id):

        self.jobs[job_id].execute()

    def _pre_exec_post_job(self, job_id):

        if not self.prePostExec is None:
            self.prePostExec.pre_exec(self,job_id)

        self.execute_job(job_id)

        if not self.prePostExec is None:
            self.prePostExec.post_exec(self,job_id)

    def execute(self, job_id_list=None):

        if job_id_list is None:
            job_id_list=range(0, len(self.jobs))

        if self.i_am_executing:
            return

        self.i_am_executing = True

        # prepare the case data
        if self.comm is None:
            comm=self.comm
            size=1
            rank=0
        else:
            comm=self.comm
            size=comm.size
            rank=comm.rank

        self.pre_run()
        job_list=[None]*len(self.jobs)

        # Check if we are running under MPI
        if size>1:

            # If we have enough processors then just execute all jobs at once according to rank
            if size>=len(job_id_list):
                for rank, job_id in enumerate(job_id_list):
                    job_list[job_id].append(rank)
                if rank<len(job_id_list):
                    self._pre_exec_post_job(job_id_list[rank])

            # We do not have enough processors so execute with a round robin
            else:
                status=MPI.Status()

                # Am I the job coordinator?
                if rank == 0:
                    task_index = 0
                    num_workers = size - 1
                    closed_workers = 0
                    while closed_workers < num_workers:
                        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                        source = status.Get_source()
                        tag = status.Get_tag()
                        if tag == tags.READY:
                            if task_index < len(job_id_list):
                                job_list[job_id_list[task_index]]=source
                                comm.send(job_id_list[task_index], dest=source, tag=tags.START)
                                task_index += 1
                            else:
                                comm.send(None, dest=source, tag=tags.EXIT)
                        elif tag == tags.DONE:
                            results = data
                        elif tag == tags.EXIT:
                            closed_workers += 1
                    comm.bcast(job_list, root=0)
                # No I am a worker...
                else:
                    while True:
                        comm.send(None, dest=0, tag=tags.READY)
                        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                        tag = status.Get_tag()
                        if tag == tags.START:
                            # Do the work here
                            self._pre_exec_post_job(task)
                            result=0
                            comm.send(result, dest=0, tag=tags.DONE)
                        elif tag == tags.EXIT:
                            break
                    comm.send(None, dest=0, tag=tags.EXIT)

                    # collect the job list (to coordinate broadcasts)
                    job_list=comm.bcast(None, root=0)


        # If not in MPI run all the jobs serially
        else:

            for job_id in job_id_list:
                self._pre_exec_post_job(job_id)
                job_list[job_id]=rank

        self.post_run(job_list)

        self.i_am_executing = False

    # set-up the communicator
    def _setup_communicators(self, comm, parent_dir):

        self.comm=comm

# this is the case runner for actually running Fused-Objects
class FUSED_MPI_ObjectCases(FUSED_MPI_Cases):

    def __init__(self, jobs=[], comm=None, prePostExec=None):
        super(FUSED_MPI_ObjectCases, self).__init__(jobs=jobs, comm=comm, prePostExec=prePostExec)

        self.sync_arg = '__downstream__'
        for job in self.jobs:
            job.set_case_runner(self)
            job.disable_external_push()

    # This will make sure that the upstream calculations are completed
    def pre_run(self):

        # open all the jobs for building
        for job in self.jobs:
            job.open_for_building_input_vector()
        # we need to pull the input so that part of the script runs in serial
        for job in self.jobs:
            job.collect_input_data()
        # open all the jobs for building
        for job in self.jobs:
            job.close_for_building_input_vector()

    # This will ad the jobs
    def add_job(self, job):

        # disable the external push
        job.disable_external_push()
        # Add the job
        FUSED_MPI_Cases.add_job(self, job)

    # This will execute the job
    def execute_job(self, job_id):

        self.jobs[job_id].update_output_data()

    # This will ensure that all the data is synchronized
    def post_run(self, job_list):

        # clear the input vector in case it has not already
        for job in self.jobs:
            job.clear_input_vector()

        # if we are not running in MPI, then just exit
        if MPI is None:
            return

        # First indicate that all the data is distributed
        for i, job in enumerate(self.jobs):
            if not job_list[i] is None:
                job.set_as_remotely_calculated(job_list[i])
                job.sync_output(self.sync_arg)

# This function will build a directory path within an MPI environment
def build_path(path):
    is_abs = path[0]=='/'
    path_split = path.split('/')
    for i in range(0, len(path_split)):
        try_cnt = 0
        test_dir = os.path.join(*(path_split[0:i+1]))
        if is_abs:
            test_dir = '/'+test_dir
        if len(test_dir)>0:
            while not os.path.isdir(test_dir):
                try:
                    os.mkdir(test_dir)
                except:
                    time.sleep(0.5)
                    try_cnt+=1
                    if try_cnt==3:
                        raise Exception('Tried 3 times to create the directory')

##
#
# These are the methods the fused object needs to work with this object in a case runner
#
# can_reload_results()            -> return true or false
# get_case_directory()            -> returns a path that differentiates the case
# get_simulation_directory()      -> returns the folder where it will write the simulation
# set_to_forced_failure()         -> this sets-up the object to skip over execution and fail
# set_to_forced_calculation()     -> this sets-up the object to calculate
# set_to_reload_results()         -> this sets-up the object to skip calculation and reload data
# reset_from_forced_failure()     -> restore object from skip execution and failing
# reset_from_forced_calculation() -> restore object from forced calculation
# reset_from_reload_results()     -> restore object from skip execution and reload data
#

# This is a class that will copy the fused-wind results from a data base so that they can be re-used
class PrePostExec_SaveReloadSimulation(object):

    def __init__(self,
                save_file_list,
                save_directory_list,
                model_extractor=None,
                fail_file='failed',
                database_folder='./database/',
                database_fail_folder=None,
                rerun_failed_jobs=False,
                doe_design_id_extractor=None,
                doe_design_directory_name = 'doe_design_',
                clear_sim_dir=False,
                link_files_on_reload=True,
                log_file_name=None
            ):
        super(PrePostExec_SaveReloadSimulation, self).__init__()

        # This is the list of files that needs to be saved
        self.save_file_list=save_file_list

        # This is the list of directories that needs to be saved
        self.save_directory_list=save_directory_list

        # This is the name of the file that is created to indicate that the object failed in the data-base
        self.fail_file = fail_file

        # This stores where the results are stored
        self.database = database_folder

        # This is the extractor function
        self.model_extractor = model_extractor

        # This is a function that can be used to extract the doe job id
        self.doe_design_id_extractor = doe_design_id_extractor

        # Set the doe design directory name
        self.doe_design_directory_name = doe_design_directory_name

        # Indicates that when re-loading old results, that the files should be linked
        self.link_files_on_reload = link_files_on_reload

        # Indicates that the simulation directory should be cleared before execution
        self.clear_sim_dir = clear_sim_dir

        # This is the log file
        self.log_file_name = log_file_name

        # This is the log file
        self.log_file = None

        # This is the folder for logging failure
        self.database_fail_folder = database_fail_folder

        # This will force the jobs, that had failed, to re-run
        self.rerun_failed_jobs = rerun_failed_jobs

        # Indicates a forced failure in the pre
        self.forced_reload = {}

        # Indicates a forced failure in the pre
        self.forced_failure = {}

        # Indicates a forced calculation in the pre
        self.forced_calculation = {}

        # Indicates that the links must be deleted
        self.delete_links = {}

    # This is the deconstructor
    def __del__(self):
        if not self.log_file is None:
            self.log_file.close()

    # this will log messages
    def _log_message(self, message):
        if not self.log_file_name is None and self.log_file is None:
            self.log_file = open(self.log_file_name, 'w')
        if not self.log_file is None:
            self.log_file.write(message+'\n')

    # this will log messages
    def _flush_log(self):
        if not self.log_file is None:
            self.log_file.flush()

    # this will test if the simulation failed
    def _has_failed(self, data_base_dir):
        self._log_message('\t\tTesting whether we have failure in directory %s'%(data_base_dir))
        if not self.fail_file is None:
            test_file = os.path.join(data_base_dir,self.fail_file)
            if os.path.isfile(test_file):
                self._log_message('\t\t\tFound the file %s, thus it failed'%(test_file))
                return True
        self._log_message('\t\t\tNo failure detected')
        return False

    # This will place a file indicating that HAWC2 has failed
    def _store_failure(self, data_base_dir):
        self._log_message('\t\tAbout to store the failure in directory %s'%(data_base_dir))
        if not self.fail_file is None:
            test_file = os.path.join(data_base_dir,self.fail_file)
            if not os.path.isfile(test_file):
                self._log_message('\t\t\tFile indicating failure is stored here %s'%(test_file))
                open(test_file, 'a').close()
                return
            self._log_message('\t\t\tFailed to write %s because it already exists'%(test_file))
        self._log_message('\t\t\tFailed to write file because no file name is given')

    # Remove the fail-flag
    def _remove_failure(self, data_base_dir):
        self._log_message('\t\tAbout to remove the fail-file in directory %s'%(data_base_dir))
        if not self.fail_file is None:
            test_file = os.path.join(data_base_dir,self.fail_file)
            if os.path.isfile(test_file):
                self._log_message('\t\t\tFile indicating failure is stored here %s'%(test_file))
                os.remove(test_file)
            else:
                self._log_message('\t\tIt seems the file does not exist %s'%(test_file))
        else:
            self._log_message('\t\tIt seems we are not configured to indicate failure')

    # this will test whether a set of files and directories exist
    def _does_data_exist(self, data_base_dir):

        self._log_message('\t\tAbout to see if data exists in directory %s'%(data_base_dir))
        for file in self.save_file_list:
            test_file = os.path.join(data_base_dir,file)
            if not os.path.isfile(test_file):
                self._log_message('\t\t\tThe file %s does NOT exist, thus exiting with false'%(test_file))
                return False
            self._log_message('\t\t\tThe file %s does exist, thus continuing to test'%(test_file))
        for dir in self.save_directory_list:
            test_dir = os.path.join(data_base_dir,dir)
            if not os.path.isdir(test_dir):
                self._log_message('\t\t\tThe directory %s does NOT exist, thus exiting with false'%(test_dir))
                return False
            self._log_message('\t\t\tThe directory %s does exist, thus continuing to test'%(test_dir))
        self._log_message('\t\t\tEverything exists, thus exiting with true')
        return True

    # This will copy all the results
    def _copy_results(self, src_dir, dst_dir):
        self._log_message('\t\tAbout to copy the results, from %s to %s'%(src_dir, dst_dir))
        # Loop through all files and directories and copy
        for file in self.save_file_list:
            src = os.path.join(src_dir,file)
            if os.path.isfile(src):
                self._log_message('\t\t\tCalling the following command: %s'%('cp -r '+src+' '+dst_dir))
                os.system('cp -r '+src+' '+dst_dir)
            else:
                self._log_message('\t\t\tCANNOT call the following command, because the source does not exist: %s'%('cp -r '+src+' '+dst_dir))
        for dir in self.save_directory_list:
            src = os.path.join(src_dir,dir)
            if os.path.isdir(src):
                self._log_message('\t\t\tCalling the following command: %s'%('cp -r '+src+' '+dst_dir))
                os.system('cp -r '+src+' '+dst_dir)
            else:
                self._log_message('\t\t\tCANNOT call the following command, because the source does not exist: %s'%('cp -r '+src+' '+dst_dir))

    # This will rsync the results
    def _rsync_directory(self, src_dir, dst_dir, with_delete_flag = False):
        self._log_message('\t\tAbout to rsync the directory with src %s to dst %s and %r'%(src_dir, dst_dir, with_delete_flag))
        if not with_delete_flag:
            if os.path.isdir(src_dir):
                comm_str='rsync -a '+src_dir+'/ '+dst_dir+'/'
                self._log_message('\t\t\tCalling the command: %s'%(comm_str))
                os.system(comm_str)
            else:
                self._log_message('\t\t\tCannot rsync the failure because the directory has not been created')
        else:
            if os.path.isdir(src_dir):
                comm_str='rsync -a --delete '+src_dir+'/ '+dst_dir+'/'
                self._log_message('\t\t\tCalling the command: %s'%(comm_str))
                os.system(comm_str)
            else:
                self._log_message('\t\t\tCannot rsync the failure because the directory has not been created')

    # This will copy all the results
    def _link_results(self, src_dir, dst_dir):
        self._log_message('\t\tAbout to link the results, from directory %s to %s'%(src_dir, dst_dir))
        #Scratch directory:
        scratch_dir = os.getcwd()
        # Loop through all files and directories and copy
        for file in self.save_file_list:
            src = os.path.join(scratch_dir,src_dir,file)
            dst = os.path.join(dst_dir,file)
            if os.path.isfile(src):
                if os.path.isfile(dst):
                    self._log_message('\t\t\tCalling the following command: %s'%('rm '+dst))
                    os.system('rm '+dst)
                self._log_message('\t\t\tCalling the following command: %s'%('ln -s -T '+src+' '+dst))
                os.system('ln -s -T '+src+' '+dst)
            else:
                self._log_message('\t\t\tCANNOT call the following command because the sources do not exist: %s'%('ln -s -T '+src+' '+dst))
        for dir in self.save_directory_list:
            src = os.path.join(scratch_dir,src_dir,dir)
            dst = os.path.join(dst_dir,dir)
            if os.path.isdir(src):
                if os.path.isdir(dst):
                    self._log_message('\t\t\tCalling the following command: %s'%('rm '+dst))
                    os.system('rm -r '+dst)
                self._log_message('\t\t\tCalling the following command: %s'%('ln -s -T '+src+' '+dst))
                os.system('ln -s -T '+src+' '+dst)
            else:
                self._log_message('\t\t\tCANNOT call the following command because the sources do not exist: %s'%('ln -s -T '+src+' '+dst))

    # This will copy all the results
    def _delink_results(self, dst_dir):
        self._log_message('\t\tAbout to de-link the results in result %s'%(dst_dir))
        # Loop through all files and directories and copy
        for file in self.save_file_list:
            dst = os.path.join(dst_dir,file)
            if os.path.islink(dst):
                self._log_message('\t\t\tCalling the following command: %s'%('rm '+dst))
                os.system('rm '+dst)
        for dir in self.save_directory_list:
            dst = os.path.join(dst_dir,dir)
            if os.path.islink(dst):
                self._log_message('\t\t\tCalling the following command: %s'%('rm '+dst))
                os.system('rm '+dst)

    # This is the pre-exec function
    def pre_exec(self, case_runner, job_id):

        self._log_message('Calling pre-exec with the following job-id %d'%(job_id))

        # Retrieve the database folder
        database=self.database
        # Retrieve the object list
        obj_list = [case_runner.jobs[job_id]]
        if not self.model_extractor is None:
            obj_list = self.model_extractor(case_runner.jobs[job_id])
        # Retrieve the doe design id from the job id or from the extractor
        doe_design_id = job_id
        if not self.doe_design_id_extractor is None:
            doe_design_id = self.doe_design_id_extractor(case_runner, job_id)
        self._log_message('\tThe doe design id is %d'%(doe_design_id))

        # Loop over all the objects and retrieve the results if they exist
        for fw_obj in obj_list:
            self._log_message('\tAbout to process object of type %s and name %s'%(str(type(fw_obj)), fw_obj.object_name))

            # Only copy results if the conditions are right
            if fw_obj.can_reload_results():
                # Get the case directory
                case_dir = fw_obj.get_case_directory()
                # build the doe directory
                doe_dir = self.doe_design_directory_name+str(doe_design_id)
                # build the data base directory
                data_base_dir = os.path.join(database,doe_dir,case_dir)
                # log the dirs
                self._log_message('\t\tThe case-dir: %s'%case_dir)
                self._log_message('\t\tThe doe-dir: %s'%doe_dir)
                self._log_message('\t\tThe database-dir: %s'%data_base_dir)

                # load flags
                has_failed = self._has_failed(data_base_dir)
                has_results = False
                if not has_failed:
                    has_results = self._does_data_exist(data_base_dir)

                # Test if object failed
                if has_failed and not self.rerun_failed_jobs:
                    self._log_message('\t\tIt seems the simulation has failed, processing it to indicate failure')
                    fw_obj.set_to_forced_failure()
                    self.forced_failure[fw_obj] = True
                # If no failure and has old results, reload those results
                elif not has_failed and has_results:
                    self._log_message('\t\tIt seems the simulation has already ran, processing it to copy over the old results')
                    # Force the object to re-load results
                    fw_obj.set_to_reload_results()
                    self.forced_reload[fw_obj] = True
                    # Get the simulation directory
                    sim_dir = fw_obj.get_simulation_directory()
                    # Build the path
                    build_path(sim_dir)
                    # retrieve the old results
                    if self.link_files_on_reload:
                        self._log_message('\t\tAbout to link from %s to %s'%(data_base_dir, sim_dir))
                        # copy the results
                        self._link_results(data_base_dir, sim_dir)
                        self.delete_links[fw_obj]=True
                    else:
                        self._log_message('\t\tAbout to copy from %s to %s'%(data_base_dir, sim_dir))
                        # copy the results
                        self._copy_results(data_base_dir, sim_dir)
                # run the job if there are no results
                elif not has_results or (has_failed and self.rerun_failed_jobs):
                    if not has_results:
                        self._log_message('\t\tForcing the simulation to re-calculate because there are no old files')
                    else:
                        self._log_message('\t\tForcing the simulation to re-calculate because we are configured to re-calculate failed results')
                    fw_obj.set_to_forced_calculation()
                    self.forced_calculation[fw_obj] = True
                    if self.clear_sim_dir:
                        # Get the simulation directory
                        sim_dir = fw_obj.get_simulation_directory()
                        # Delete any old files
                        if os.path.isdir(sim_dir):
                            self._log_message('\t\tClearing the simulation directory "%s" with the command "%s"'%(sim_dir, 'rm -rf %s/*'%(sim_dir)))
                            os.system('rm -rf %s/*'%(sim_dir))
                        else:
                            self._log_message('\t\tThe simulation directory "%s" does not exist, nothing to clear'%(sim_dir))
            else:
                self._log_message('\t\tUsing old results disabled, so there is nothing to do')

        # Flush the log
        self._flush_log()

    # This is the post-exec function
    def post_exec(self, case_runner, job_id):

        self._log_message('Calling post-exec with the following job-id %d'%(job_id))

        # Retrieve the database folder
        database=self.database
        database_fail_folder=self.database_fail_folder
        # Retrieve the object list
        obj_list = [case_runner.jobs[job_id]]
        if not self.model_extractor is None:
            obj_list = self.model_extractor(case_runner.jobs[job_id])
        # Retrieve the doe design id from the job id or from the extractor
        doe_design_id = job_id
        if not self.doe_design_id_extractor is None:
            doe_design_id = self.doe_design_id_extractor(case_runner, job_id)
        self._log_message('\tThe doe design id is %d'%(doe_design_id))
        #Scratch directory:
        scratch_dir = os.getcwd()
        self._log_message('\tThe scratch directory is %s'%(scratch_dir))

        # Loop over all the objects and retrieve the results if they exist
        for fw_obj in obj_list:
            self._log_message('\tAbout to process fused-wind object of type %s and name %s'%(str(type(fw_obj)), fw_obj.object_name))

            # get dir list for the write directory
            case_dir = fw_obj.get_case_directory()
            # build the doe directory
            doe_dir = self.doe_design_directory_name+str(doe_design_id)
            # build the data base directory
            data_base_dir = os.path.join(database,doe_dir,case_dir)
            if not database_fail_folder is None:
                data_base_fail_dir = os.path.join(database_fail_folder,doe_dir,case_dir)
            else:
                data_base_fail_dir = None
            # Build the simulation directory
            sim_dir = fw_obj.get_simulation_directory()
            self._log_message('\t\tThe case-dir: %s'%case_dir)
            self._log_message('\t\tThe doe-dir: %s'%doe_dir)
            self._log_message('\t\tThe database-dir: %s'%data_base_dir)
            # Only copy results if the conditions are right
            if fw_obj.succeeded:
                self._log_message('\t\tIt seems that object has succeeded')
                # Copy only if there is no data
                if not self._does_data_exist(data_base_dir):
                    self._log_message('\t\tIt seems that the database data does not exist')
                    # build path to the database
                    build_path(data_base_dir)
                    # copy the results
                    self._log_message('\t\tCopying results from %s to %s'%(sim_dir, data_base_dir))
                    self._copy_results(sim_dir, data_base_dir)
                elif self._has_failed(data_base_dir):
                    self._log_message('\t\tIt seems that the old job failed, but a re-run succeeded')
                    # remove the fail flag
                    self._remove_failure(data_base_dir)
                    # copy the results
                    self._log_message('\t\tCopying results from %s to %s'%(sim_dir, data_base_dir))
                    self._copy_results(sim_dir, data_base_dir)
                else:
                    self._log_message('\t\tIt seems that the database data already exists')
            else:
                self._log_message('\t\tIt seems that the object has failed')
                # copy failure only if data doesn't exist and fail file doesn't exist
                has_failed = self._has_failed(data_base_dir)
                has_results = self._does_data_exist(data_base_dir)
                if not has_failed and not has_results:
                    # build path to the database
                    build_path(data_base_dir)
                    # copy the results
                    self._log_message('\t\tCopying results from %s to %s'%(sim_dir, data_base_dir))
                    self._copy_results(sim_dir, data_base_dir)
                    # Create the failure flag
                    self._log_message('\t\tStoring failure at %s'%(data_base_dir))
                    self._store_failure(data_base_dir)
                # Send a message on the status flags
                if has_failed:
                    self._log_message('\t\tIt seems that the database data already indicates failure')
                if has_results:
                    self._log_message('\t\tIt seems that the database data already exists')
                # rsync the failure
                if not data_base_fail_dir is None:
                    self._log_message('\t\tThe data_base_fail_dir is set to %s'%(data_base_fail_dir))
                    if os.path.isdir(data_base_fail_dir):
                        self._log_message('\t\tOld results exist, synchronizing over these results')
                        self._rsync_directory(sim_dir, data_base_fail_dir, with_delete_flag=True)
                    else:
                        self._log_message('\t\tNo old results exist, synchronizing over fresh results')
                        build_path(data_base_fail_dir)
                        self._rsync_directory(sim_dir, data_base_fail_dir, with_delete_flag=False)

            # If forced failure, then clean-up
            if fw_obj in self.forced_failure and self.forced_failure[fw_obj]:
                fw_obj.reset_from_forced_failure()
                self.forced_failure[fw_obj] = False
                self._log_message('\t\tRestoring configuration after forcing a failure')

            # If was forced to execute, then reset the old results
            if fw_obj in self.forced_calculation and self.forced_calculation[fw_obj]:
                fw_obj.reset_from_forced_calculation()
                self.forced_calculation[fw_obj] = False
                self._log_message('\t\tRestoring configuration after forcing an execution')

            # If object was forced to re-load, then restore that
            if fw_obj in self.forced_reload and self.forced_reload[fw_obj]:
                fw_obj.reset_from_reload_results()
                self.forced_reload[fw_obj] = False
                self._log_message('\t\tRestoring configuration after forcing a reload')

            # delete the link if needed
            if fw_obj in self.delete_links and self.delete_links[fw_obj]:
                self.delete_links[fw_obj]=False
                self._delink_results(sim_dir)
                self._log_message('\t\tDeleting the links from re-loading old results')

        # Flush the log
        self._flush_log()

# This is a pre-post-exec that runs multiple versions
class PrePostExec_MultiPrePost(object):

    # Lets build the init
    def __init__(self, pre_post_list):
        super(PrePostExec_MultiPrePost, self).__init__()

        self.pre_post_list = pre_post_list

    # This is the pre version
    def pre_exec(self, case_runner, job_id):

        for obj in self.pre_post_list:
            obj.pre_exec(case_runner, job_id)

    # This is the post version
    def post_exec(self, case_runner, job_id):

        for obj in self.pre_post_list:
            obj.post_exec(case_runner, job_id)

