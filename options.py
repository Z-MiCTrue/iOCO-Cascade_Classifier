class Parameters:
    def __init__(self):
        # 正样本目标尺度, 也为检测时的尺度
        self.aim_w = 28
        self.aim_h = 28
        # 训练参数
        self.numStages = 20
        self.featureType = 'LBP'
        self.minHitRate = 0.996
        self.maxFalseAlarmRate = 0.12
        self.mode = 'ALL'
