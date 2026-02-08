import multiprocessing as mp
import constants
import processes

if __name__ == "__main__":
    with mp.Manager() as manager:
        model_queue = manager.Queue(1)
        data_queue = manager.Queue(constants.queue_size)
        env = mp.Process(target=processes.collect, args=(data_queue, model_queue), kwargs=constants.collect_params, daemon=True)
        opt = mp.Process(target=processes.optimize, args=(data_queue, model_queue), kwargs=constants.optimize_params, daemon=False)
        env.start()
        opt.start()
        opt.join()
        env.terminate()
