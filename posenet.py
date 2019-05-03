from network import Network

class GoogLeNet(Network):
    def setup(self):
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, name='conv1')
             .max_pool(3, 3, 2, 2, name='pool1')
             #???.max_pool(3, 3, 2, 2, name='pool1', padding='VALID')
             # tensorflow and caffe different: local_size = 2*depth_radius + 1
             # alpha的定义caffe要在实际的系数上乘以local_size 
             .lrn(2, 2e-05, 0.75, name='norm1')
             #2019 .conv(1, 1, 64, 1, 1, name='reduction2')
             .conv(1, 1, 64, 1, 1, name='reduction2', padding='VALID')
             .conv(3, 3, 192, 1, 1, name='conv2')
             .lrn(2, 2e-05, 0.75, name='norm2')
             .max_pool(3, 3, 2, 2, name='pool2')
             #???.max_pool(3, 3, 2, 2, name='pool2', padding='VALID')
             #2019 .conv(1, 1, 96, 1, 1, name='icp1_reduction1')
             .conv(1, 1, 96, 1, 1, name='icp1_reduction1', padding='VALID')
             .conv(3, 3, 128, 1, 1, name='icp1_out1'))

        (self.feed('pool2')
             #2019 .conv(1, 1, 16, 1, 1, name='icp1_reduction2')
             .conv(1, 1, 16, 1, 1, name='icp1_reduction2', padding='VALID')
             .conv(5, 5, 32, 1, 1, name='icp1_out2'))

        (self.feed('pool2')
             .max_pool(3, 3, 1, 1, name='icp1_pool')
             #2019 .conv(1, 1, 32, 1, 1, name='icp1_out3'))
             .conv(1, 1, 32, 1, 1, name='icp1_out3', padding='VALID'))

        (self.feed('pool2')
             #2019 .conv(1, 1, 64, 1, 1, name='icp1_out0'))
             .conv(1, 1, 64, 1, 1, name='icp1_out0', padding='VALID'))

        (self.feed('icp1_out0', 
                   'icp1_out1', 
                   'icp1_out2', 
                   'icp1_out3')
             #2019 .concat(3, name='icp2_in')
             .concat(axis=3, name='icp2_in')
             #2019 .conv(1, 1, 128, 1, 1, name='icp2_reduction1')
             .conv(1, 1, 128, 1, 1, name='icp2_reduction1', padding='VALID')
             .conv(3, 3, 192, 1, 1, name='icp2_out1'))

        (self.feed('icp2_in')
             #2019 .conv(1, 1, 32, 1, 1, name='icp2_reduction2')
             .conv(1, 1, 32, 1, 1, name='icp2_reduction2', padding='VALID')
             .conv(5, 5, 96, 1, 1, name='icp2_out2'))

        (self.feed('icp2_in')
             .max_pool(3, 3, 1, 1, name='icp2_pool')
             #2019 .conv(1, 1, 64, 1, 1, name='icp2_out3'))
             .conv(1, 1, 64, 1, 1, name='icp2_out3', padding='VALID'))

        (self.feed('icp2_in')
             #2019 .conv(1, 1, 128, 1, 1, name='icp2_out0'))
             .conv(1, 1, 128, 1, 1, name='icp2_out0', padding='VALID'))

        (self.feed('icp2_out0', 
                   'icp2_out1', 
                   'icp2_out2', 
                   'icp2_out3')
             #2019 .concat(3, name='icp2_out')
             .concat(axis=3, name='icp2_out')
             .max_pool(3, 3, 2, 2, name='icp3_in')
             #???.max_pool(3, 3, 2, 2, name='icp3_in', padding='VALID')
             #2019 .conv(1, 1, 96, 1, 1, name='icp3_reduction1')
             .conv(1, 1, 96, 1, 1, name='icp3_reduction1', padding='VALID')
             .conv(3, 3, 208, 1, 1, name='icp3_out1'))

        (self.feed('icp3_in')
             #2019 .conv(1, 1, 16, 1, 1, name='icp3_reduction2')
             .conv(1, 1, 16, 1, 1, name='icp3_reduction2', padding='VALID')
             .conv(5, 5, 48, 1, 1, name='icp3_out2'))

        (self.feed('icp3_in')
             .max_pool(3, 3, 1, 1, name='icp3_pool')
             #2019 .conv(1, 1, 64, 1, 1, name='icp3_out3'))
             .conv(1, 1, 64, 1, 1, name='icp3_out3', padding='VALID'))

        (self.feed('icp3_in')
             #2019 .conv(1, 1, 192, 1, 1, name='icp3_out0'))
             .conv(1, 1, 192, 1, 1, name='icp3_out0', padding='VALID'))

        (self.feed('icp3_out0', 
                   'icp3_out1', 
                   'icp3_out2', 
                   'icp3_out3')
             #2019 .concat(3, name='icp3_out')
             .concat(axis=3, name='icp3_out')
             .avg_pool(5, 5, 3, 3, padding='VALID', name='cls1_pool')
             #2019 .conv(1, 1, 128, 1, 1, name='cls1_reduction_pose')
             .conv(1, 1, 128, 1, 1, name='cls1_reduction_pose', padding='VALID')
             .fc(1024, name='cls1_fc1_pose')
             #2019 
             .dropout(0.7, name='cls1_drop')
             .fc(3, relu=False, name='cls1_fc_pose_xyz'))

        #2019 (self.feed('cls1_fc1_pose')
        (self.feed('cls1_drop')
             .fc(4, relu=False, name='cls1_fc_pose_wpqr'))

        (self.feed('icp3_out')
             #2019 .conv(1, 1, 112, 1, 1, name='icp4_reduction1')
             .conv(1, 1, 112, 1, 1, name='icp4_reduction1', padding='VALID')
             .conv(3, 3, 224, 1, 1, name='icp4_out1'))

        (self.feed('icp3_out')
             #2019 .conv(1, 1, 24, 1, 1, name='icp4_reduction2')
             .conv(1, 1, 24, 1, 1, name='icp4_reduction2', padding='VALID')
             .conv(5, 5, 64, 1, 1, name='icp4_out2'))

        (self.feed('icp3_out')
             .max_pool(3, 3, 1, 1, name='icp4_pool')
             #2019 .conv(1, 1, 64, 1, 1, name='icp4_out3'))
             .conv(1, 1, 64, 1, 1, name='icp4_out3', padding='VALID'))

        (self.feed('icp3_out')
             #2019 .conv(1, 1, 160, 1, 1, name='icp4_out0'))
             .conv(1, 1, 160, 1, 1, name='icp4_out0', padding='VALID'))

        (self.feed('icp4_out0', 
                   'icp4_out1', 
                   'icp4_out2', 
                   'icp4_out3')
             #2019 .concat(3, name='icp4_out')
             .concat(axis=3, name='icp4_out')
             #2019 .conv(1, 1, 128, 1, 1, name='icp5_reduction1')
             .conv(1, 1, 128, 1, 1, name='icp5_reduction1', padding='VALID')
             .conv(3, 3, 256, 1, 1, name='icp5_out1'))

        (self.feed('icp4_out')
             #2019 .conv(1, 1, 24, 1, 1, name='icp5_reduction2')
             .conv(1, 1, 24, 1, 1, name='icp5_reduction2', padding='VALID')
             .conv(5, 5, 64, 1, 1, name='icp5_out2'))

        (self.feed('icp4_out')
             .max_pool(3, 3, 1, 1, name='icp5_pool')
             #2019 .conv(1, 1, 64, 1, 1, name='icp5_out3'))
             .conv(1, 1, 64, 1, 1, name='icp5_out3', padding='VALID'))

        (self.feed('icp4_out')
             #2019 .conv(1, 1, 128, 1, 1, name='icp5_out0'))
             .conv(1, 1, 128, 1, 1, name='icp5_out0', padding='VALID'))

        (self.feed('icp5_out0', 
                   'icp5_out1', 
                   'icp5_out2', 
                   'icp5_out3')
             #2019 .concat(3, name='icp5_out')
             .concat(axis=3, name='icp5_out')
             #.conv(1, 1, 144, 1, 1, name='icp6_reduction1')
             .conv(1, 1, 144, 1, 1, name='icp6_reduction1', padding='VALID')
             .conv(3, 3, 288, 1, 1, name='icp6_out1'))

        (self.feed('icp5_out')
             #2019 .conv(1, 1, 32, 1, 1, name='icp6_reduction2')
             .conv(1, 1, 32, 1, 1, name='icp6_reduction2', padding='VALID')
             .conv(5, 5, 64, 1, 1, name='icp6_out2'))

        (self.feed('icp5_out')
             .max_pool(3, 3, 1, 1, name='icp6_pool')
             #2019 .conv(1, 1, 64, 1, 1, name='icp6_out3'))
             .conv(1, 1, 64, 1, 1, name='icp6_out3', padding='VALID'))

        (self.feed('icp5_out')
             #2019 .conv(1, 1, 112, 1, 1, name='icp6_out0'))
             .conv(1, 1, 112, 1, 1, name='icp6_out0', padding='VALID'))

        (self.feed('icp6_out0', 
                   'icp6_out1', 
                   'icp6_out2', 
                   'icp6_out3')
             #2019 .concat(3, name='icp6_out')
             .concat(axis=3, name='icp6_out')
             .avg_pool(5, 5, 3, 3, padding='VALID', name='cls2_pool')
             #2019 .conv(1, 1, 128, 1, 1, name='cls2_reduction_pose')
             .conv(1, 1, 128, 1, 1, name='cls2_reduction_pose', padding='VALID')
             .fc(1024, name='cls2_fc1')
             #2019 
             .dropout(0.7, name='cls2_drop')
             .fc(3, relu=False, name='cls2_fc_pose_xyz'))

        # 2019 (self.feed('cls2_fc1')
        (self.feed('cls2_drop')
             .fc(4, relu=False, name='cls2_fc_pose_wpqr'))

        (self.feed('icp6_out')
             #2019 .conv(1, 1, 160, 1, 1, name='icp7_reduction1')
             .conv(1, 1, 160, 1, 1, name='icp7_reduction1', padding='VALID')
             .conv(3, 3, 320, 1, 1, name='icp7_out1'))

        (self.feed('icp6_out')
             #2019 .conv(1, 1, 32, 1, 1, name='icp7_reduction2')
             .conv(1, 1, 32, 1, 1, name='icp7_reduction2', padding='VALID')
             .conv(5, 5, 128, 1, 1, name='icp7_out2'))

        (self.feed('icp6_out')
             .max_pool(3, 3, 1, 1, name='icp7_pool')
             #2019 .conv(1, 1, 128, 1, 1, name='icp7_out3'))
             .conv(1, 1, 128, 1, 1, name='icp7_out3', padding='VALID'))

        (self.feed('icp6_out')
             .conv(1, 1, 256, 1, 1, name='icp7_out0', padding='VALID'))
             #2019 .conv(1, 1, 256, 1, 1, name='icp7_out0'))

        (self.feed('icp7_out0', 
                   'icp7_out1', 
                   'icp7_out2', 
                   'icp7_out3')
             #2019 .concat(3, name='icp7_out')
             .concat(axis=3, name='icp7_out')
             .max_pool(3, 3, 2, 2, name='icp8_in')
             #.max_pool(3, 3, 2, 2, name='icp8_in', padding='VALID')
             #2019 .conv(1, 1, 160, 1, 1, name='icp8_reduction1')
             .conv(1, 1, 160, 1, 1, name='icp8_reduction1', padding='VALID')
             .conv(3, 3, 320, 1, 1, name='icp8_out1'))

        (self.feed('icp8_in')
             #2019 .conv(1, 1, 32, 1, 1, name='icp8_reduction2')
             .conv(1, 1, 32, 1, 1, name='icp8_reduction2', padding='VALID')
             .conv(5, 5, 128, 1, 1, name='icp8_out2'))

        (self.feed('icp8_in')
             .max_pool(3, 3, 1, 1, name='icp8_pool')
             #2019 .conv(1, 1, 128, 1, 1, name='icp8_out3'))
             .conv(1, 1, 128, 1, 1, name='icp8_out3', padding='VALID'))

        (self.feed('icp8_in')
             #2019 .conv(1, 1, 256, 1, 1, name='icp8_out0'))
             .conv(1, 1, 256, 1, 1, name='icp8_out0', padding='VALID'))

        (self.feed('icp8_out0', 
                   'icp8_out1', 
                   'icp8_out2', 
                   'icp8_out3')
             #2019 .concat(3, name='icp8_out')
             .concat(axis=3, name='icp8_out')
             #2019 .conv(1, 1, 192, 1, 1, name='icp9_reduction1')
             .conv(1, 1, 192, 1, 1, name='icp9_reduction1', padding='VALID')
             .conv(3, 3, 384, 1, 1, name='icp9_out1'))

        (self.feed('icp8_out')
             #2019 .conv(1, 1, 48, 1, 1, name='icp9_reduction2')
             .conv(1, 1, 48, 1, 1, name='icp9_reduction2', padding='VALID')
             .conv(5, 5, 128, 1, 1, name='icp9_out2'))

        (self.feed('icp8_out')
             .max_pool(3, 3, 1, 1, name='icp9_pool')
             #2019 .conv(1, 1, 128, 1, 1, name='icp9_out3'))
             .conv(1, 1, 128, 1, 1, name='icp9_out3', padding='VALID'))

        (self.feed('icp8_out')
             #2019 .conv(1, 1, 384, 1, 1, name='icp9_out0'))
             .conv(1, 1, 384, 1, 1, name='icp9_out0', padding='VALID'))

        (self.feed('icp9_out0', 
                   'icp9_out1', 
                   'icp9_out2', 
                   'icp9_out3')
             #2019 .concat(3, name='icp9_out')
             .concat(axis=3, name='icp9_out')
             .avg_pool(7, 7, 1, 1, padding='VALID', name='cls3_pool')
             #.avg_pool(7, 7, 1, 1, padding='SAME', name='cls3_pool')
             .fc(2048, name='cls3_fc1_pose')
             #2019 0.5 
             .dropout(0.7, name='cls3_drop')
             .fc(3, relu=False, name='cls3_fc_pose_xyz'))

        #2019 (self.feed('cls3_fc1_pose')
        (self.feed('cls3_drop')
             .fc(4, relu=False, name='cls3_fc_pose_wpqr'))