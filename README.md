# ca_widget (Widget + Auto-updating HF Chatbot)

这个仓库现在包含两部分：

1) **前端小组件（widget）**：用 iframe 把 Hugging Face Space 的聊天页面嵌进学校网站（原有代码保持不变）。
2) **自动更新的数据管线（crawler → HF Space）**：GitHub Actions 定时爬取学校官网内容，生成 `hf_space/sources/*.txt`，并**自动上传**到 Hugging Face Space（不需要你再手动复制数据到 app.py）。

---

## 你需要在 GitHub 仓库里设置的 Secrets

进入 GitHub Repo → Settings → Secrets and variables → Actions → New repository secret：

- `HF_TOKEN`：Hugging Face token（对你的 Space 有写权限）
- `HF_SPACE_ID`：你的 Space Repo ID，例如：`abyss2009/cheshire-academy-chatbot`
- （可选）`BASE_URL`：要爬取的网站根目录，默认 `https://www.cheshireacademy.org/`
- （可选）`MAX_PAGES`：最多抓取页面数，默认 `600`

> 注意：OpenAI 的 `OPENAI_API_KEY` **不放在 GitHub**，而是放在 Hugging Face Space → Settings → Secrets 里。

---

## Hugging Face Space 需要的 Secrets / Variables

Space → Settings → Secrets：

- `OPENAI_API_KEY`
- （可选）`OPENAI_MODEL`（默认 `gpt-4o-mini`）
- （可选）`OPENAI_TEMPERATURE`（默认 `0.2`）
- （可选）`RAG_TOP_K`（默认 `4`）
- （可选）`RAG_MAX_CHARS`（默认 `4500`）

---

## 自动更新是怎么工作的？

GitHub Actions 工作流：`.github/workflows/update_hf_space.yml`

- **每 6 小时**运行一次（你可以改 cron）
- 运行 `crawler/crawl_site.py` 生成 `hf_space/sources/_manifest.json` + 多个 `.txt`
- 运行 `crawler/upload_to_hf.py` 把 `hf_space/` 整个文件夹上传到你的 Space（覆盖更新）

---

## 本地测试（可选）

### 1) 测试爬虫
```bash
pip install -r crawler/requirements.txt
python crawler/crawl_site.py
```

### 2) 测试 HF Space（本地）
```bash
pip install -r hf_space/requirements.txt
OPENAI_API_KEY=你的key python hf_space/app.py
```

---

## Widget 入口

widget 默认 iframe 指向：
`https://abyss2009-chatgpt-chatbot.hf.space/`

你可以在 `src/index.js` 里改 iframe 链接（如果你的 Space 域名不同）。
