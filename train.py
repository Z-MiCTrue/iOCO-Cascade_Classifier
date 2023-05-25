import cv2
import os

from options import Parameters


class train_xml:
    def __init__(self, para):
        self.work_path = os.path.split(os.path.realpath(__file__))[0]
        self.para = para
        # 样本数记录数据
        self.pos_num = 0
        self.neg_num = 0

    def generate_txt(self):
        # 生成正样本txt数据说明以及规范数据格式
        write_str = ''
        for root, dirs, files in os.walk('./dataset/pos'):  # 工作目录, 子目录, 文件
            for img_name in files:
                img = cv2.imread(f'./dataset/pos/{img_name}')
                h, w = img.shape[:2]
                if w < self.para.aim_w or h < self.para.aim_h:  # 尺寸过小的样本将被舍弃
                    print('log: positive sample discarded')
                else:
                    write_str += f'{self.work_path}\\dataset\\pos\\{img_name} 1 0 0 {w} {h}\n'
                    self.pos_num += 1
        with open('./opencv_toolkit/pos.txt', 'w') as result_file:
            result_file.write(write_str)
        # 生成负样本txt数据说明以及规范数据格式
        write_str = ''
        for root, dirs, files in os.walk('./dataset/neg'):  # 工作目录, 子目录, 文件
            for img_name in files:
                write_str += f'{self.work_path}\\dataset\\neg\\{img_name}\n'
                self.neg_num += 1
        with open('./opencv_toolkit/neg.txt', 'w') as result_file:
            result_file.write(write_str[:-1])
        # 生成正样本vec数据文件以及打印当前命令
        cmd = f'opencv_createsamples.exe -info pos.txt ' \
              f'-vec pos.vec -num {self.pos_num} -w {self.para.aim_w} -h {self.para.aim_h}'
        print(f'command: {cmd}')
        os.chdir('./opencv_toolkit/')
        os.system(cmd)
        os.chdir(self.work_path)

    def start_train(self, batch_size=48):
        # 这里只是类似于batch size, 即为每一级分类器所用到的正样本数, 设置的数量要小于总体正样本数, 太大会报错
        # 生成批处理bat数据文件
        if worker.pos_num <= batch_size:
            pos_use = worker.pos_num
        else:
            pos_use = batch_size
        if 3 * pos_use >= self.neg_num:
            neg_use = self.neg_num
        else:
            neg_use = 3 * pos_use
        write_str = f'opencv_traincascade -data xml -vec pos.vec -bg neg.txt ' \
                    f'-numStages {self.para.numStages} ' \
                    f'-featureType {self.para.featureType} ' \
                    f'-minHitRate {self.para.minHitRate} ' \
                    f'-maxFalseAlarmRate {self.para.maxFalseAlarmRate} ' \
                    f'-mode {self.para.mode} ' \
                    f'-w {self.para.aim_w} -h {self.para.aim_h} -numPos {pos_use} -numNeg {neg_use}\n\npause'
        with open('./opencv_toolkit/start_train.bat', 'w') as result_file:
            result_file.write(write_str)
        # 输入1开始训练
        continue_switch = int(input('\nFile writing completed. Continue? (0 & 1)\necho: '))
        if continue_switch:
            cmd = 'start_train.bat'
            print(f'\ncommand: {cmd}')
            os.chdir('./opencv_toolkit/')
            os.system(cmd)
            os.chdir(self.work_path)
        else:
            print('over')


if __name__ == '__main__':
    parameters = Parameters()
    worker = train_xml(parameters)

    worker.generate_txt()
    worker.start_train(batch_size=36)
