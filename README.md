# 比起原UP的差异
 - 增加了思维链的输出，默认是思维链接thinkingBudget=2048，这个参数设成0则不进行思维链。
 - 增加了Function Call处理，传入和回调答案
   
```
  if (req.thinking) {
    if (req.thinking.type === "enabled" && req.thinking.budget_tokens) {
      cfg.thinkingConfig = {
        thinkingBudget: req.thinking.budget_tokens,
        includeThoughts: true,
      };
    } else if (
      typeof req.thinking === "object" &&
      req.thinking.thinkingBudget !== undefined
    ) {
      cfg.thinkingConfig = {
        thinkingBudget: req.thinking.thinkingBudget,
        includeThoughts: true,
      };
    }
  } else {
    // Set default thinking configuration if not specified
    cfg.thinkingConfig = {
      thinkingBudget: 2048,
      includeThoughts: true,
    };
  }
```
 - 我用起来没bug，有什么问题尽量自己处理。代码不复杂，搞不定找AI


# Gemini 2.0 代理

使用 Deno 免费代理 Google Gemini，国内直连，不限地区/网络环境，打开即用。

免费使用 Gemini 地表最强大的 100万 Token 上下文模型。

兼容的 OpenAI 格式，可对接 AI 编程，接入ChatBox、Cherry Studio、Cursor、Cline 等 AI 客户端。

## 分享

- [Midjourney API](https://github.com/trueai-org/midjourney-proxy)：市面上最强大，完全免费开源的免费绘图平台。
- [MDrive](https://github.com/trueai-org/mdrive)：阿里云盘官方 API 授权的自动同步和备份工具，支持挂载到本地磁盘。
- [Grok](https://github.com/trueai-org/grok)：使用 Deno 免费代理马斯克 Grok3，免登录，国内直连，不限地区/不限网络/不限设备，打开即用。

## 本地部署（推荐）

> 由于 Deno 封锁问题，推荐使用本地部署到国外的服务器（例如：新加坡）

```bash
# 1. 克隆项目（如果还没有）
git clone https://github.com/trueai-org/gemini.git
cd gemini

# 2. 给部署脚本执行权限
chmod +x deploy.sh

# 3. 运行部署
./deploy.sh
```

## Deno 部署

> Bilibili 视频教程：<https://www.bilibili.com/video/BV1DDA3eEEYn/>

1. 免费创建一个 Gemini API Key [https://aistudio.google.com](https://aistudio.google.com/app/apikey)
1. 点击 [Fork](https://github.com/trueai-org/gemini/fork) 本项目（万分感谢帮助点个 `Star`）
2. 登录/注册 Deno https://dash.deno.com/
3. 点击创建项目 https://dash.deno.com/new_project
4. 选择此项目，填写项目名字（分配域名）
5. 部署 Entrypoint 填写 `src/deno_index.ts` 其他字段留空 
6. 点击 **Deploy Project**
7. 部署成功后获得域名，可以作为Chat API的代理使用。
8. 下载安装 **[Cherry Studio](https://cherry-ai.com/)** -> 设置 -> 添加模型服务 -> 输入域名和 Token -> 添加模型 -> 开启 AI 会话。

## Deno 调试

Windows 安装 Deno:
> irm https://deno.land/install.ps1 | iex

Mac/Linux 安装 Deno:
> curl -fsSL https://deno.land/install.sh | sh

启动项目：

> cd gemini

> deno run --allow-net --allow-read src/deno_index.ts

## 鸣谢

- https://github.com/tech-shrimp/gemini-playground
- https://github.com/PublicAffairs/openai-gemini
