import paddle
import yaml
from paddlex.ppocr.data.rec.dataset import LMDBDataSet
from paddlex.ppocr.modeling.architectures import RecModel
from paddlex.ppocr.postprocess.rec_postprocess import CTCLabelDecode
from paddle.io import DataLoader

# 读取YAML配置
with open("src/config/rec_anti_fake.yml", "r") as f:
    config = yaml.safe_load(f)

# 数据加载
train_dataset = LMDBDataSet(
    config["Train"]["dataset"]["data_dir"],
    config["Train"]["dataset"]["label_file_list"],
    transforms=config["Train"]["dataset"]["transforms"]
)
train_dataloader = DataLoader(train_dataset, batch_size=config["Train"]["loader"]["batch_size_per_card"], shuffle=True)

# 构建模型
model = RecModel(config["Architecture"])

# 优化器
optimizer = paddle.optimizer.Adam(
    learning_rate=config["Optimizer"]["lr"]["learning_rate"],
    parameters=model.parameters()
)

# 训练循环
for epoch in range(config["Global"]["epoch_num"]):
    for batch_id, data in enumerate(train_dataloader):
        images, labels = data
        preds = model(images)
        loss = model.loss(preds, labels)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if batch_id % 10 == 0:
            print(f"Epoch [{epoch}/{config['Global']['epoch_num']}], Batch [{batch_id}], Loss: {loss.numpy()[0]}")
