
try:
    from mpi4py import MPI
except:
    print('It seems that we are not able to import MPI')
    MPI = None

def enum(*sequential, **named):
    from site import print_trace_now; print_trace_now()
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Define MPI message tags
tags = enum('READY', 'DONE', 'EXIT', 'START')

print('MIMC some debug code here')
log_file_path = '/home/mimc/python_environments/Smart_Tip_SBO_Py3_MPI/tkba_hawc2_example/case_runner_log_rank_%d'%(MPI.COMM_WORLD.rank)
#log_file = open(log_file_path, 'w')

# This is a case iterator that will run under MPI

# This is a generic case runner
class FUSED_MPI_Cases(object):

    def __init__(self, jobs=[], comm=None, preExec=None, postExec=None):
        from site import print_trace_now; print_trace_now()
        super(FUSED_MPI_Cases, self).__init__()

        self.comm=comm
        if not MPI is None and self.comm==None:
            self.comm=MPI.COMM_WORLD

        self.jobs = jobs
        self.i_am_executing = False
        self.preExec = preExec
        self.postExec = postExec

    def add_job(self, job):
        from site import print_trace_now; print_trace_now()

        self.jobs.append(job)

    def get_job_count(self):
        from site import print_trace_now; print_trace_now()

        return len(self.jobs)

    def get_job(self, at_id):
        from site import print_trace_now; print_trace_now()

        return self.jobs[at_id]

    def pre_run(self):
        from site import print_trace_now; print_trace_now()
        pass

    def post_run(self, job_list):
        from site import print_trace_now; print_trace_now()
        # Note job list stores the rank that completed a given job
        pass

    def execute_job(self, job_id):
        from site import print_trace_now; print_trace_now()

        self.jobs[job_id].execute()

    def _pre_exec_post_job(self, job_id):
        from site import print_trace_now; print_trace_now()

        if not self.preExec is None:
            self.preExec(self,job_id)

        self.execute_job(job_id)

        if not self.postExec is None:
            self.postExec(self, job_id)

    def execute(self):
        from site import print_trace_now; print_trace_now()

        if self.i_am_executing:
            return

        self.i_am_executing = True

        # Check if we are running under MPI
        if not self.comm is None and self.comm.size>1:

            self.pre_run()

            # prepare the case data
            comm=self.comm
            size=comm.size
            rank=comm.rank
            job_list=[]

            # If we have enough processors then just execute all jobs at once according to rank
            if size>=len(self.jobs):
                for i in range(len(self.jobs)):
                    job_list.append(i)
                if rank<len(self.jobs):
                    self._pre_exec_post_job(rank)

            # We do not have enough processors so execute with a round robin
            else:
                status=MPI.Status()

                # Am I the job coordinator?
                if rank == 0:
                    task_index = 0
                    num_workers = size - 1
                    closed_workers = 0
                    print('CASE_RUNNER ROOT working with %d workers, about to start the case-runner'%(num_workers), file=open(log_file_path, 'a'))
                    while closed_workers < num_workers:
                        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                        source = status.Get_source()
                        tag = status.Get_tag()
                        if tag == tags.READY:
                            if task_index < len(self.jobs):
                                print('CASE_RUNNER ROOT acknowledges that rank %d is ready, sending job %d to execute'%(source, task_index), file=open(log_file_path, 'a'))
                                job_list.append(source)
                                comm.send(task_index, dest=source, tag=tags.START)
                                task_index += 1
                            else:
                                print('CASE_RUNNER ROOT acknowledges that rank %d is ready, no jobs left, sending exit instruction'%(source), file=open(log_file_path, 'a'))
                                comm.send(None, dest=source, tag=tags.EXIT)
                        elif tag == tags.DONE:
                            print('CASE_RUNNER ROOT acknowledges that rank %d is done'%(source), file=open(log_file_path, 'a'))
                            results = data
                        elif tag == tags.EXIT:
                            closed_workers += 1
                            print('CASE_RUNNER ROOT acknowledges that rank %d is exiting, number of closed workers is %d'%(source, closed_workers), file=open(log_file_path, 'a'))
                    print('CASE_RUNNER ROOT %d Resume serial'%(rank), file=open(log_file_path, 'a'))
                    comm.bcast(job_list, root=0)
                # No I am a worker...
                else:
                    print('CASE_RUNNER RANK %d about to start working'%(rank), file=open(log_file_path, 'a'))
                    while True:
                        print('CASE_RUNNER RANK %d about to ask for a job'%(rank), file=open(log_file_path, 'a'))
                        comm.send(None, dest=0, tag=tags.READY)
                        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                        tag = status.Get_tag()
                        if tag == tags.START:
                            # Do the work here
                            print('CASE_RUNNER RANK %d about to execute job %d'%(rank, task), file=open(log_file_path, 'a'))
                            self._pre_exec_post_job(task)
                            result=0
                            print('CASE_RUNNER RANK %d completed job %d'%(rank, task), file=open(log_file_path, 'a'))
                            comm.send(result, dest=0, tag=tags.DONE)
                        elif tag == tags.EXIT:
                            print('CASE_RUNNER RANK %d About to exit'%(rank), file=open(log_file_path, 'a'))
                            break
                    comm.send(None, dest=0, tag=tags.EXIT)

                    # collect the job list (to coordinate broadcasts)
                    print('CASE_RUNNER RANK %d Resume serial'%(rank), file=open(log_file_path, 'a'))
                    job_list=comm.bcast(None, root=0)

            self.post_run(job_list)

        # If not in MPI run all the jobs serially
        else:

            for job_id in range(0, len(self.jobs)):
                self._pre_exec_post_job(job_id)

        self.i_am_executing = False

    # set-up the communicator
    def _setup_communicators(self, comm, parent_dir):
        from site import print_trace_now; print_trace_now()

        self.comm=comm

# this is the case runner for actually running Fused-Objects
class FUSED_MPI_ObjectCases(FUSED_MPI_Cases):

    def __init__(self, jobs=[], comm=None, preExec=None, postExec=None):
        from site import print_trace_now; print_trace_now()
        super(FUSED_MPI_ObjectCases, self).__init__(jobs=jobs, comm=comm, preExec=preExec, postExec=postExec)

        self.sync_arg = '__downstream__'
        for job in self.jobs:
            job.set_case_runner(self)

    # This will make sure that the upstream calculations are completed
    def pre_run(self):
        from site import print_trace_now; print_trace_now()

        # we need to pull the input so that part of the script runs in serial
        for job in self.jobs:
            job.collect_input_data()

    # This will execute the job
    def execute_job(self, job_id):
        from site import print_trace_now; print_trace_now()

        self.jobs[job_id].update_output_data()

    # This will ensure that all the data is synchronized
    def post_run(self, job_list):
        from site import print_trace_now; print_trace_now()

        # First indicate that all the data is distributed
        for i, job in enumerate(self.jobs):
            job.set_as_remotely_calculated(job_list[i])
            job.sync_output(self.sync_arg)

