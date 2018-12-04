
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

    def add_job(self, job):

        self.jobs.append(job)

    def get_job_count(self):

        return len(self.jobs)

    def get_job(self, at_id):

        return self.jobs[at_id]

    def pre_run(self):
        pass

    def post_run(self, job_list):
        pass

    def execute_job(self, job_id):

        result = self.jobs[job_id].execute()
        return result

    def execute(self):

        # Check if we are running under MPI
        if not self.comm is None and self.comm.size>1:
            self.pre_run()

            # prepare the case data
            comm=self.comm
            size=comm.size
            rank=comm.rank
            status=MPI.Status()
            job_list=[]

            # If we have enough processors then just execute all jobs at once according to rank
            if size>=len(self.jobs):
                for i in range(len(self.jobs)):
                    job_list.append(i)
                if rank<len(self.jobs):
                    result = self.execute_job(rank)
                    print('{} doing his job'.format(rank))

                #Preparing the case where the result is not None and should be gathered:
                if rank == 0 and not result is None:
                    results = [result]
                    num_workers = size-1
                    closed_workers = 0
                    while closed_workers < num_workers:
                        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                        results.append(data)
                        closed_workers +=1
                    return results
                elif not result is None:
                    comm.send(result, dest=0, tag=tags.DONE)

            # We do not have enough processors so execute with a round robin
            else:
                status=MPI.Status()

                # Am I the job coordinator?
                if rank == 0:
                    task_index = 0
                    num_workers = size - 1
                    closed_workers = 0
                    result_index = 0
                    results = [None]*len(self.jobs)
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
                            results[result_index] = data
                            result_index += 1
                        elif tag == tags.EXIT:
                            closed_workers += 1
                    comm.bcast(job_list, root=0)
                    return results
                # No I am a worker...
                else:
                    while True:
                        comm.send(None, dest=0, tag=tags.READY)
                        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                        tag = status.Get_tag()
                        if tag == tags.START:
                            # Do the work here
                            result = self.execute_job(task)
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

    # set-up the communicator
    def _setup_communicators(self, comm, parent_dir):

        self.comm=comm

# this is the case runner for actually running Fused-Objects
class FUSED_MPI_ObjectCases(FUSED_MPI_Cases):

    def __init__(self, jobs=[], comm=None):
        super(FUSED_MPI_ObjectCases, self).__init__(jobs, comm)

    def pre_run(self):

        # we need to pull the input so that part of the script runs in serial
        for job in jobs:
            job._build_input_vector()

    def execute_job(self, job_id):

        self.jobs[job_id].get_output_value()

    def post_run(self, job_list):

        comm=self.comm
        size=comm.size
        rank=comm.rank

        # now perform broadcasts so all processes have the same data
        for i, src in enumerate(job_list):
            if rank==src:
                my_output = self.jobs[i].get_output_value()
                send_keys = list(my_output.keys())
                # broadcast the keys
                comm.bcast(send_keys, root=src)
                # loop over the results
                for k in send_keys:
                    comm.bcast(my_output[k], root=src)
            else:
                # broadcast the keys
                recv_keys = comm.bcast(None, root=src)
                # loop over the results
                for k in recv_keys:
                    output_value = comm.bcast(None, root=src)
                    self.jobs[i]._set_output_value(k, output_value)

            if rank==0:
                import pdb;pdb.set_trace()

class FUSED_MPI_DataSetCases(FUSED_MPI_Cases):

    def __init__(self, data_set_object, jobs=[], comm=None):
        super(FUSED_MPI_DataSetCases, self).__init__(jobs, comm)

    def post_run(self, job_list):

        comm=self.comm
        size=comm.size
        rank=comm.rank

        numpy_arrays = data_set_object.get_numpy_array()
        
        for i, src in enumerate(job_list):
            if rank == src:
                my_output = self.jobs[i].get_output()
                comm.bcast(my_output, root=src)
            else:
                output = comm.bcast(None, root=src)
                self.jobs[i].set_output(output)


        #Here is needed a method that collects everything

