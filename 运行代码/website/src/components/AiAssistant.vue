<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'

/* ====== 拖拽相关 ====== */
const position = ref({ x: 0, y: 0 })
const isDragging = ref(false)
const dragOffset = ref({ x: 0, y: 0 })
const wiggle = ref(false)
const wasDragged = ref(false)
const startPointer = ref({ x: 0, y: 0 })

/* ====== 聊天相关 ====== */
const showChat = ref(false)
const chatMessages = ref([
  { role: 'assistant', content: '你好！我是广西气象AI助手，可以为你解答关于广西气象数据分析的问题，比如降水趋势、气温变化、干旱监测等。请问有什么可以帮你的？' }
])
const userInput = ref('')
const isLoading = ref(false)
const chatBody = ref(null)

const DEEPSEEK_API_KEY = import.meta.env.VITE_DEEPSEEK_API_KEY || 'YOUR_API_KEY_HERE'
const DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions'

const assetsBase = document.baseURI.replace(/#.*$/, '')
const aiImg = assetsBase + 'ai_assistant.png'
const chatBg = assetsBase + 'chat-header-bg.png'

const SYSTEM_PROMPT = `你是广西气象数据分析平台的AI助手。你精通气象学、气候统计和数据分析，尤其熟悉广西壮族自治区的气象特征。
你能帮助用户理解以下内容：
- 广西8个代表城市的气象数据分析结果
- 时间序列分解与趋势检验（Mann-Kendall、Sen斜率）
- 干旱监测指标（SPI、SPEI）
- 极端气候事件检测
- 深度学习预测模型（TCN、LSTM、GRU、Transformer）
- 多元回归分析与特征重要性
- 聚类分析与城市气候分类
请用简洁专业的中文回答用户问题。`

async function sendMessage() {
  const text = userInput.value.trim()
  if (!text || isLoading.value) return

  chatMessages.value.push({ role: 'user', content: text })
  userInput.value = ''
  isLoading.value = true
  await nextTick()
  scrollToBottom()

  try {
    const messages = [
      { role: 'system', content: SYSTEM_PROMPT },
      ...chatMessages.value
    ]

    const response = await fetch(DEEPSEEK_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${DEEPSEEK_API_KEY}`
      },
      body: JSON.stringify({
        model: 'deepseek-chat',
        messages,
        temperature: 0.7,
        max_tokens: 2048
      })
    })

    if (!response.ok) {
      throw new Error(`API请求失败: ${response.status}`)
    }

    const data = await response.json()
    const reply = data.choices?.[0]?.message?.content || '抱歉，暂时无法获取回复。'
    chatMessages.value.push({ role: 'assistant', content: reply })
  } catch (err) {
    chatMessages.value.push({ role: 'assistant', content: `抱歉，AI服务暂时不可用：${err.message}。请稍后重试。` })
  } finally {
    isLoading.value = false
    await nextTick()
    scrollToBottom()
  }
}

function scrollToBottom() {
  if (chatBody.value) {
    chatBody.value.scrollTop = chatBody.value.scrollHeight
  }
}

function handleKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
}

function clearChat() {
  chatMessages.value = [
    { role: 'assistant', content: '你好！我是广西气象AI助手，可以为你解答关于广西气象数据分析的问题。请问有什么可以帮你的？' }
  ]
}

function formatMessage(text) {
  return text
    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>')
}

/* ====== 拖拽逻辑 ====== */
function onPointerDown(e) {
  if (e.target.closest('.ai-chat') || e.target.closest('.ai-bubble-close')) return
  isDragging.value = true
  wasDragged.value = false
  startPointer.value = { x: e.clientX, y: e.clientY }
  dragOffset.value = {
    x: e.clientX - position.value.x,
    y: e.clientY - position.value.y,
  }
  e.preventDefault()
}

function onPointerMove(e) {
  if (!isDragging.value) return
  const dx = e.clientX - startPointer.value.x
  const dy = e.clientY - startPointer.value.y
  if (Math.abs(dx) > 5 || Math.abs(dy) > 5) {
    wasDragged.value = true
  }
  position.value = {
    x: e.clientX - dragOffset.value.x,
    y: e.clientY - dragOffset.value.y,
  }
}

function onPointerUp() {
  if (isDragging.value) {
    isDragging.value = false
    wiggle.value = true
    setTimeout(() => { wiggle.value = false }, 600)
  }
}

function onClick() {
  if (wasDragged.value) return
  wiggle.value = true
  setTimeout(() => { wiggle.value = false }, 600)
  showChat.value = !showChat.value
}

onMounted(() => {
  const w = window.innerWidth
  const h = window.innerHeight
  position.value = { x: w - 160, y: h - 230 }

  window.addEventListener('pointermove', onPointerMove)
  window.addEventListener('pointerup', onPointerUp)
})

onUnmounted(() => {
  window.removeEventListener('pointermove', onPointerMove)
  window.removeEventListener('pointerup', onPointerUp)
})
</script>

<template>
  <div
    class="ai-assistant"
    :class="{ dragging: isDragging, wiggle }"
    :style="{ left: position.x + 'px', top: position.y + 'px' }"
  >
    <!-- 聊天窗口（居中弹窗） -->
    <Teleport to="body">
      <Transition name="chat-overlay">
        <div v-if="showChat" class="ai-chat-overlay" @click.self="showChat = false">
          <Transition name="chat-window">
            <div v-if="showChat" class="ai-chat">
        <!-- 头部 -->
        <div
          class="ai-chat-header"
          :style="{ backgroundImage: 'url(' + chatBg + ')', backgroundSize: 'cover', backgroundPosition: 'center' }"
        >
          <div class="ai-chat-header-left">
            <img :src="aiImg" alt="AI" class="ai-chat-avatar" />
            <div>
              <div class="ai-chat-title">气象AI助手</div>
              <div class="ai-chat-status">在线 · DeepSeek</div>
            </div>
          </div>
          <div class="ai-chat-header-actions">
            <button class="ai-chat-btn-icon" @click="clearChat" title="清空对话">🗑</button>
            <button class="ai-chat-btn-icon" @click="showChat = false" title="关闭">✕</button>
          </div>
        </div>

        <!-- 消息列表 -->
        <div ref="chatBody" class="ai-chat-body">
          <div
            v-for="(msg, i) in chatMessages"
            :key="i"
            class="ai-chat-message"
            :class="msg.role"
          >
            <div v-if="msg.role === 'assistant'" class="ai-msg-avatar">
              <img :src="aiImg" alt="AI" />
            </div>
            <div class="ai-msg-bubble">
              <div class="ai-msg-text" v-html="formatMessage(msg.content)"></div>
            </div>
          </div>
          <!-- 加载动画 -->
          <div v-if="isLoading" class="ai-chat-message assistant">
            <div class="ai-msg-avatar">
              <img :src="aiImg" alt="AI" />
            </div>
            <div class="ai-msg-bubble">
              <div class="ai-typing">
                <span></span><span></span><span></span>
              </div>
            </div>
          </div>
        </div>

        <!-- 输入区 -->
        <div class="ai-chat-footer">
          <textarea
            v-model="userInput"
            placeholder="输入你的问题…"
            rows="1"
            class="ai-chat-input"
            @keydown="handleKeydown"
          ></textarea>
          <button
            class="ai-chat-send"
            :disabled="!userInput.trim() || isLoading"
            @click="sendMessage"
          >
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </div>
      </div>
    </Transition>
        </div>
      </Transition>
    </Teleport>

    <!-- 助手主体 -->
    <div
      class="ai-avatar"
      :class="{ 'ai-avatar-hover': !isDragging }"
      @pointerdown="onPointerDown"
      @click="onClick"
    >
      <img :src="aiImg" alt="AI气象助手" draggable="false" />
      <div class="ai-glow"></div>
    </div>
  </div>
</template>

<style scoped>
.ai-assistant {
  position: fixed;
  z-index: 9999;
  pointer-events: none;
  user-select: none;
}

.ai-assistant.dragging {
  transition: none !important;
  will-change: left, top;
}

/* ====== 头像 ====== */
.ai-avatar {
  width: 120px;
  height: 160px;
  pointer-events: auto;
  cursor: pointer;
  position: relative;
  filter: drop-shadow(0 6px 16px rgba(255, 165, 0, 0.2));
  transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
  animation: ai-float 3s ease-in-out infinite;
}

.ai-avatar img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  pointer-events: none;
  transition: transform 0.3s ease;
}

.ai-avatar-hover:hover img {
  transform: scale(1.08) rotate(-3deg);
}

.ai-assistant.dragging .ai-avatar {
  cursor: grabbing;
  transform: scale(1.05);
}

.ai-assistant.wiggle .ai-avatar {
  animation: ai-wiggle 0.5s ease-in-out;
}

/* ====== 呼吸光环 ====== */
.ai-glow {
  position: absolute;
  bottom: -4px;
  left: 50%;
  transform: translateX(-50%);
  width: 90px;
  height: 18px;
  background: radial-gradient(ellipse, rgba(255, 165, 0, 0.25) 0%, transparent 70%);
  border-radius: 50%;
  animation: ai-glow-pulse 3s ease-in-out infinite;
  pointer-events: none;
}

/* ====== 遮罩层 ====== */
.ai-chat-overlay {
  position: fixed;
  inset: 0;
  z-index: 10000;
  background: rgba(0, 0, 0, 0.35);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: auto;
}

/* ====== 聊天窗口 ====== */
.ai-chat {
  width: 420px;
  height: 600px;
  background: #ffffff;
  border-radius: 20px;
  box-shadow: 0 8px 40px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(0, 0, 0, 0.05);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  pointer-events: auto;
}

/* 头部 */
.ai-chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px;
  color: white;
  position: relative;
}

.ai-chat-header::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(30, 64, 120, 0.7), rgba(29, 78, 216, 0.55));
}

.ai-chat-header-left {
  display: flex;
  align-items: center;
  gap: 10px;
  position: relative;
  z-index: 1;
}

.ai-chat-header-actions {
  display: flex;
  gap: 4px;
  position: relative;
  z-index: 1;
}

.ai-chat-avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  object-fit: cover;
  border: 2px solid rgba(255, 255, 255, 0.4);
}

.ai-chat-title {
  font-size: 14px;
  font-weight: 600;
}

.ai-chat-status {
  font-size: 11px;
  opacity: 0.8;
}

.ai-chat-btn-icon {
  background: rgba(255, 255, 255, 0.15);
  border: none;
  color: white;
  width: 30px;
  height: 30px;
  border-radius: 8px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  transition: background 0.2s;
}

.ai-chat-btn-icon:hover {
  background: rgba(255, 255, 255, 0.3);
}

/* 消息区 */
.ai-chat-body {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  background: #f8fafc;
}

.ai-chat-body::-webkit-scrollbar {
  width: 4px;
}
.ai-chat-body::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 2px;
}

.ai-chat-message {
  display: flex;
  gap: 8px;
  max-width: 92%;
}

.ai-chat-message.user {
  align-self: flex-end;
  flex-direction: row-reverse;
}

.ai-msg-avatar {
  width: 28px;
  height: 28px;
  flex-shrink: 0;
}

.ai-msg-avatar img {
  width: 100%;
  height: 100%;
  border-radius: 50%;
  object-fit: cover;
}

.ai-msg-bubble {
  padding: 10px 14px;
  border-radius: 14px;
  font-size: 13px;
  line-height: 1.7;
  word-break: break-word;
}

.assistant .ai-msg-bubble {
  background: white;
  color: #1e293b;
  border: 1px solid #e2e8f0;
  border-bottom-left-radius: 4px;
}

.user .ai-msg-bubble {
  background: linear-gradient(135deg, #3b82f6, #1d4ed8);
  color: white;
  border-bottom-right-radius: 4px;
}

.ai-msg-text :deep(pre) {
  background: #1e293b;
  color: #e2e8f0;
  padding: 10px 12px;
  border-radius: 8px;
  overflow-x: auto;
  font-size: 12px;
  margin: 8px 0;
}

.ai-msg-text :deep(.inline-code) {
  background: #f1f5f9;
  color: #e11d48;
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 12px;
}

/* 打字动画 */
.ai-typing {
  display: flex;
  gap: 4px;
  padding: 4px 0;
}

.ai-typing span {
  width: 7px;
  height: 7px;
  background: #94a3b8;
  border-radius: 50%;
  animation: typing-bounce 1.4s ease-in-out infinite;
}
.ai-typing span:nth-child(2) { animation-delay: 0.2s; }
.ai-typing span:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing-bounce {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
  30% { transform: translateY(-6px); opacity: 1; }
}

/* 输入区 */
.ai-chat-footer {
  display: flex;
  align-items: flex-end;
  gap: 8px;
  padding: 12px 14px;
  border-top: 1px solid #e2e8f0;
  background: white;
}

.ai-chat-input {
  flex: 1;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 10px 14px;
  font-size: 13px;
  font-family: "Noto Sans SC", "Microsoft YaHei", sans-serif;
  resize: none;
  outline: none;
  transition: border-color 0.2s;
  line-height: 1.5;
  max-height: 80px;
  color: #1e293b;
  background: #f8fafc;
}

.ai-chat-input:focus {
  border-color: #3b82f6;
  background: white;
}

.ai-chat-send {
  width: 38px;
  height: 38px;
  flex-shrink: 0;
  border: none;
  border-radius: 12px;
  background: linear-gradient(135deg, #3b82f6, #1d4ed8);
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: opacity 0.2s, transform 0.15s;
}

.ai-chat-send:hover:not(:disabled) {
  transform: scale(1.05);
}

.ai-chat-send:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* ====== 动画 ====== */
@keyframes ai-float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-8px); }
}

@keyframes ai-wiggle {
  0% { transform: rotate(0deg); }
  15% { transform: rotate(-6deg); }
  30% { transform: rotate(5deg); }
  45% { transform: rotate(-3deg); }
  60% { transform: rotate(2deg); }
  75% { transform: rotate(-1deg); }
  100% { transform: rotate(0deg); }
}

@keyframes ai-glow-pulse {
  0%, 100% { opacity: 0.5; transform: translateX(-50%) scaleX(1); }
  50% { opacity: 1; transform: translateX(-50%) scaleX(1.2); }
}

/* ====== 遮罩层过渡 ====== */
.chat-overlay-enter-active {
  transition: opacity 0.25s ease;
}
.chat-overlay-leave-active {
  transition: opacity 0.2s ease;
}
.chat-overlay-enter-from,
.chat-overlay-leave-to {
  opacity: 0;
}

/* ====== 窗口过渡 ====== */
.chat-window-enter-active {
  animation: chat-in 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.chat-window-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}
.chat-window-leave-to {
  opacity: 0;
  transform: scale(0.9) translateY(10px);
}

@keyframes chat-in {
  0% { opacity: 0; transform: scale(0.85) translateY(20px); }
  100% { opacity: 1; transform: scale(1) translateY(0); }
}

/* ====== 响应式 ====== */
@media (max-width: 480px) {
  .ai-chat {
    width: calc(100vw - 32px);
    height: 70vh;
  }
}
</style>
