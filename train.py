# Import the converted model's class
import numpy as np
import random
import tensorflow as tf
#from posenet import GoogLeNet as PoseNet
import cv2
from tqdm import tqdm
from vgg_pp import VGG16 as PoseNet_pp

# 201903 Variable conv1/weights/Adam/ already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:
tf.reset_default_graph()

#batch_size = 75
batch_size = 32
max_iterations = 30000
#max_iterations = 10000

# Set this path to your dataset directory
#directory = 'path_to_datasets/KingsCollege/'
#directory = 'E:/PoseNet/KingsCollege/'
directory = 'E:/PoseNet/heads/'
#directory = 'E:/PoseNet/ShopFacade/'
#dataset = 'dataset_train.txt'
dataset = 'heads_train.txt'

class datasource(object):
    def __init__(self, images, poses):
        self.images = images
        self.poses = poses

def centeredCrop(img, output_side_length):
    height, width, depth = img.shape
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = output_side_length * height / width
    else:
        new_width = output_side_length * width / height
    height_offset = (new_height - output_side_length) / 2
    width_offset = (new_width - output_side_length) / 2
    #print (width_offset)
    cropped_img = img[int(height_offset):int(height_offset) + output_side_length,
                      int(width_offset):int(width_offset) + output_side_length]
    return cropped_img

def preprocess(images):
    images_out = [] #final result
    #Resize and crop and compute mean!
    images_cropped = []
    for i in tqdm(range(len(images))):
        X = cv2.imread(images[i])
        X = cv2.resize(X, (455, 256))
        X = centeredCrop(X, 224)
        images_cropped.append(X)
    #compute images mean
    N = 0
    mean = np.zeros((1, 3, 224, 224))
    for X in tqdm(images_cropped):
        #---
        #X = np.transpose(X,(2,0,1))
        #mean[0][0] += X[:,:,0]
        #mean[0][1] += X[:,:,1]
        #mean[0][2] += X[:,:,2]
        # --- 201903 need to check
        X = np.transpose(X,(2,0,1))
        mean[0][0] += X[0,:,:]
        mean[0][1] += X[1,:,:]
        mean[0][2] += X[2,:,:]
        N += 1
    mean[0] /= N
    #Subtract mean from all images
    for X in tqdm(images_cropped):
        X = np.transpose(X,(2,0,1))
        X = X - mean
        X = np.squeeze(X)
        X = np.transpose(X, (1,2,0))
        images_out.append(X)
    return images_out

def get_data():
    poses = []
    images = []

    with open(directory+dataset) as f:
        next(f)  # skip the 3 header lines
        next(f)
        next(f)
        for line in f:
            fname, p0,p1,p2,p3,p4,p5,p6 = line.split()
            p0 = float(p0)
            p1 = float(p1)
            p2 = float(p2)
            p3 = float(p3)
            p4 = float(p4)
            p5 = float(p5)
            p6 = float(p6)
            poses.append((p0,p1,p2,p3,p4,p5,p6))
            images.append(directory+fname)
    images = preprocess(images)
    return datasource(images, poses)

def gen_data(source):
    while True:
        indices = range(len(source.images))
        #201903 TypeError: 'range' object does not support item assignment
        trainingSet = list(indices)
        random.shuffle(trainingSet)
        for i in trainingSet:
        #random.shuffle(indices)
        #for i in indices:
            image = source.images[i]
            pose_x = source.poses[i][0:3]
            pose_q = source.poses[i][3:7]
            yield image, pose_x, pose_q

def gen_data_batch(source):
    data_gen = gen_data(source)
    while True:
        image_batch = []
        pose_x_batch = []
        pose_q_batch = []
        for _ in range(batch_size):
            image, pose_x, pose_q = next(data_gen)
            image_batch.append(image)
            pose_x_batch.append(pose_x)
            pose_q_batch.append(pose_q)
        yield np.array(image_batch), np.array(pose_x_batch), np.array(pose_q_batch)


def main():
    images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    poses_x = tf.placeholder(tf.float32, [batch_size, 3])
    poses_q = tf.placeholder(tf.float32, [batch_size, 4])
    datasource = get_data()

    #net = PoseNet({'data': images})
#
#    p1_x = net.layers['cls1_fc_pose_xyz']
#    p1_q = net.layers['cls1_fc_pose_wpqr']
#    p2_x = net.layers['cls2_fc_pose_xyz']
#    p2_q = net.layers['cls2_fc_pose_wpqr']
#    p3_x = net.layers['cls3_fc_pose_xyz']
#    p3_q = net.layers['cls3_fc_pose_wpqr']

    #2019 
    vgg_beta = 100
    vgg_net = PoseNet_pp({'data': images})
    vgg_x = vgg_net.layers['fc9_pose_xyz']
    vgg_q = vgg_net.layers['fc9_pose_wpqr']
    translation_loss = tf.sqrt(tf.nn.l2_loss(tf.subtract(vgg_x, poses_x)))
    rotation_loss = tf.sqrt(tf.nn.l2_loss(tf.subtract(vgg_q, poses_q)))
    vgg_loss = translation_loss + vgg_beta * rotation_loss
    vgg_opt = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.9, beta2=0.999, epsilon=0.00000001, use_locking=False, name='Adam').minimize(vgg_loss)
    
    #2019 
    #beta = 500
    beta = 500
#    l1_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_x, poses_x)))) * 0.3
    #2019 l1_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_q, poses_q)))) * 150
#    l1_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1_q, poses_q)))) * beta*0.3
#    l2_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_x, poses_x)))) * 0.3
    #2019 l2_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_q, poses_q)))) * 150
#    l2_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p2_q, poses_q)))) * beta*0.3
#    l3_x = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3_x, poses_x)))) * 1
#    l3_q = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3_q, poses_q)))) * beta

#    loss = l1_x + l1_q + l2_x + l2_q + l3_x + l3_q
    #opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=0.00000001, use_locking=False, name='Adam').minimize(loss)
    # 201904 from paper?
#    opt = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.9, beta2=0.999, epsilon=0.00000001, use_locking=False, name='Adam').minimize(loss)
    #opt = tf.train.AdagradOptimizer(learning_rate=0.0001,initial_accumulator_value=0.1,use_locking=False,name='Adagrad').minimize(loss)
    #opt = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9,use_locking=False,name='Momentum',use_nesterov=False).minimize(loss)
    # # MyAdamW is a new class
    #MyAdamW = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
    ## Create a MyAdamW object
    #opt = MyAdamW(weight_decay=0.5, learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=0.00000001, use_locking=False).minimize(loss)
    # sess.run(optimizer.minimize(loss, decay_variables=[var1, var2]))

    # Set GPU options
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6833)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    #outputFile = "PoseNet.ckpt"
    #201903 ValueError: Parent directory of PoseNet.ckpt doesn't exist, can't save.
    outputFile = "PoseNet_scenes7-heads-vgg01.ckpt"
    #outputFile = "PoseNet_KC-v05.ckpt"
    path = "E:/PoseNet/trained_model/scenes7_heads/"
    #path = "E:/PoseNet/trained_model/KC/"

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # Load the data
        sess.run(init)
        #net.load('posenet.npy', sess)
        # 201903 changed
#        net.load('C:\\Users\\iambx\\Documents\\code_AnacondaProjects\\Github-py3_PoseNet_Tensorflow\\weights\\posenet.npy', sess)
        # 2019 for coninue training
        #saver.restore(sess, path + 'PoseNet_KC-v01.ckpt')

        data_gen = gen_data_batch(datasource)
        for i in range(max_iterations):
            np_images, np_poses_x, np_poses_q = next(data_gen)
            feed = {images: np_images, poses_x: np_poses_x, poses_q: np_poses_q}

            sess.run(vgg_opt, feed_dict=feed)
            np_loss = sess.run(vgg_loss, feed_dict=feed)
            #loss_dx, loss_dq = sess.run([l3_x, l3_q], feed_dict=feed)
            loss_dx, loss_dq = sess.run([translation_loss, rotation_loss], feed_dict=feed)
            if i % 20 == 0:
                print("iteration: " + str(i) + "\n\t" + "Loss is: " + str(np_loss))
                print ("m:", loss_dx)
                print ("degree: ", loss_dq)
            if i % 5000 == 0:
                saver.save(sess, path + outputFile)
                print("Intermediate file saved at: " + path + outputFile)
        saver.save(sess, path + outputFile)
        print("Intermediate file saved at: " + path + outputFile)


if __name__ == '__main__':
    main()
