import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np

import datasets
from positional_embeddings import PositionalEmbedding


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        # GELU 激活函数，和其他激活函数有什么不同？
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        # 此处是将自身和 过block 之后的数据相加？有什么好处？
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, 
                 hidden_size: int = 128, 
                 hidden_layers: int = 3, 
                 emb_size: int = 128,
                 time_emb: str = "sinusoidal", 
                 input_emb: str = "sinusoidal"):
        super().__init__()

        # 先用三个 postional embedding？
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        # 此处的 \ 代表换行，那么 layer 又代表什么参数？
        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        
        # 采用列表的方式来实现一个 layer
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        
        # 所以 hidden layers 对应的就是 block
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        
        # 最后应该是接一个输出层 变换到2维  
        layers.append(nn.Linear(hidden_size, 2))
        # 最后将所有的 Linear，block转换到 Sequential 对象中
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        # 相当于 x，y 坐标分别作为 embedding？
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        # 额外输入一个参数作为 timeembedding
        t_emb = self.time_mlp(t)
        # 在最后一维上将三个 embedding 拼接上
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        # 经过 mlp 处理
        x = self.joint_mlp(x)
        return x


class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):
        # 总的前向 diffusion step
        self.num_timesteps = num_timesteps
        
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2
        
        # 计算出 alpha，此处是后续跟 noise schedule 相关的一系列计算
        self.alphas = 1.0 - self.betas
        # alphas_cumprod 的每一项都是前i项 alpha 的连乘，是后面一系列变量的基础
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        # 此处的 pad 是什么意思？
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        # 输入 x_t,t,noise 来得到 x0
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        # 通过两个系数来得到 x0
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        # 通过 x_0,x_t,t 来得到 x_t-1 的均值
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        # 这里应该是默认为固定方差
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        # 此处的 tensor 的 clip 是什么意思？
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        # 用模型预测出的数值作为 noise 
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        # 再用 x_0_pred，输入的 x_t，t 来得到均值
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)
        
        # 这里采用公式造一个方差出来，为什么还要用到noise？
        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        # 把均值和方差加起来，然后作为返回值？
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        # 输入 x_0,noise,t 来得到 x_t？
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    # 选择图形
    parser.add_argument("--dataset", type=str, default="dino", choices=["circle", "dino", "line", "moons"])
    # 训练参数
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    
    # diffusion 参数
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    
    # 模型超参数
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_images_step", type=int, default=1)
    config = parser.parse_args()

    # 将 scatter 数据作为 dataloader 载入
    dataset = datasets.get_dataset(config.dataset)
    dataloader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)

    # 初始化模型
    model = MLP(
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        time_emb=config.time_embedding,
        input_emb=config.input_embedding)

    # 初始化 diffusion 公式和 noise 的 schedule
    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule)

    # 简单的 AdamW，AdamW 好在哪里？
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    global_step = 0
    # 此处的 frame 是干什么的？
    frames = []
    losses = []
    print("Training model...")
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            # 此处的 batch 就应该是 minibatch 的数据 [batch,x,y]
            batch = batch[0]
            # 每一步都要随机初始化 noise？
            noise = torch.randn(batch.shape)
            # 随机给 minibatch 中的数据设置 timestep
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (batch.shape[0],)
            ).long()

            # 采用公式得到 x_t
            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            # 模型输出 noise t-1, 这一步的 noise 到底代表什么意思？
            noise_pred = model(noisy, timesteps)
            # loss 采用 mse loss
            loss = F.mse_loss(noise_pred, noise)
            loss.backward(loss)

            # 此处为什么要采用 clip norm
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 清除这一步的梯度，不妨碍后面的训练
            optimizer.zero_grad()

            progress_bar.update(1)
            # 此处的 loss 是需要 detach 的吗？ 
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()

        if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
            # generate data with the model to later visualize the learning process
            model.eval()
            # 这里的 randn 是什么用法？
            sample = torch.randn(config.eval_batch_size, 2)
            # 这里就是生成 1000-1 的所有 step 了
            timesteps = list(range(len(noise_scheduler)))[::-1]
            for i, t in enumerate(tqdm(timesteps)):
                # 此处应该是将 t 重复 minibatch 次
                t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long()
                with torch.no_grad():
                    # 生成 noise
                    residual = model(sample, t)
                # 这里返回的 sample 是均值方差相加的结果
                sample = noise_scheduler.step(residual, t[0], sample)
            # frames 就是保存的生成的图片
            frames.append(sample.numpy())

    # 将实验结果的保存路径设置为 exps/
    print("Saving model...")
    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    # 保存模型参数
    torch.save(model.state_dict(), f"{outdir}/model.pth")

    print("Saving images...")
    imgdir = f"{outdir}/images"
    os.makedirs(imgdir, exist_ok=True)
    # 此处 np.stack 是什么用法
    frames = np.stack(frames)
    xmin, xmax = -6, 6
    ymin, ymax = -6, 6
    
    for i, frame in enumerate(frames):
        plt.figure(figsize=(10, 10))
        # frame 是 x,y 坐标
        plt.scatter(frame[:, 0], frame[:, 1])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig(f"{imgdir}/{i:04}.png")
        plt.close()

    # 还将 loss 保存在 outdir
    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))

    # 框架的坐标也直接保存
    print("Saving frames...")
    np.save(f"{outdir}/frames.npy", frames)
