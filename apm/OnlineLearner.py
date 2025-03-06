class OnlineLearner:
    def __init__(self, model, buffer_size=1000):
        self.model = model
        self.buffer = []
        self.buffer_size = buffer_size

    def on_new_query(self, query):
        # 实时生成预测
        state = self.encode(query)
        action = self.model.select_expert(state)
        pred = self.execute_action(action, query)

        # 异步收集反馈
        self.buffer.append((state, action, pred))
        if len(self.buffer) >= self.buffer_size:
            self.update_model()

    def update_model(self):
        #todo 从数据库获取真实标签
        labels = get_groundtruth(self.buffer)
        rewards = calculate_rewards(self.buffer, labels)

        # 执行PPO更新
        self.ppo_trainer.update(self.buffer, rewards)
        self.buffer.clear()