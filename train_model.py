import caffe

import config

caffe.set_device(config.train_gpu_id)
caffe.set_mode_gpu()

solver = caffe.SGDSolver(config.train_prototxt)
solver.net.copy_from(config.pretrained_model)
solver.solve()
