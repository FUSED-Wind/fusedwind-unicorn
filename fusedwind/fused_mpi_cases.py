
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

