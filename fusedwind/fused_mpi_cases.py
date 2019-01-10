
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

    def __init__(self, jobs=[], comm=None):

        super(FUSED_MPI_Cases, self).__init__()

        self.comm=comm
        if not MPI is None and self.comm==None:
            self.comm=MPI.COMM_WORLD

        self.jobs = jobs
        self.i_am_executing = False

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

    def execute(self):

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
                    self.execute_job(rank)

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
                            if task_index < len(self.jobs):
                                job_list.append(source)
                                comm.send(task_index, dest=source, tag=tags.START)
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
                            self.execute_job(task)
                            result=0
                            comm.send(result, dest=0, tag=tags.DONE)
                        elif tag == tags.EXIT:
                            break
                    comm.send(None, dest=0, tag=tags.EXIT)

                    # collect the job list (to coordinate broadcasts)
                    job_list=comm.bcast(None, root=0)

            self.post_run(job_list)

        # If not in MPI run all the jobs serially
        else:

            for job_id in range(0, len(self.jobs)):
                self.execute_job(job_id)

        self.i_am_executing = False

    # set-up the communicator
    def _setup_communicators(self, comm, parent_dir):

        self.comm=comm

# this is the case runner for actually running Fused-Objects
class FUSED_MPI_ObjectCases(FUSED_MPI_Cases):

    def __init__(self, jobs=[], comm=None, preExec=None, postExec=None):
        super(FUSED_MPI_ObjectCases, self).__init__(jobs, comm)

        self.sync_arg = '__downstream__'
        for job in self.jobs:
            job.set_case_runner(self)

        self.preExec = preExec
        self.postExec = postExec

    # This will make sure that the upstream calculations are completed
    def pre_run(self):

        # we need to pull the input so that part of the script runs in serial
        for job in self.jobs:
            job.collect_input_data()

    # This will execute the job
    def execute_job(self, job_id):
        self.preExec(self,job_id)

        self.jobs[job_id].update_output_data()
        
        self.postExec(self, job_id)

    # This will ensure that all the data is synchronized
    def post_run(self, job_list):

        # First indicate that all the data is distributed
        for i, job in enumerate(self.jobs):
            job.set_as_remotely_calculated(job_list[i])
            job.sync_output(self.sync_arg)

