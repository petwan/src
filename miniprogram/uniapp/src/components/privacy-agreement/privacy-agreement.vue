<template>
  <!-- 根容器，通过v-if控制显示 -->
  <view v-if="show" class="privacy-container">
    <!-- 遮罩层，阻止下层交互 -->
    <view class="mask" @tap="onMaskTap"></view>

    <!-- 主要内容卡片 -->
    <view class="content-card">
      <!-- 标题区域 -->
      <view class="header">
        <text class="title">隐私保护指引</text>
        <text class="subtitle">请仔细阅读以下内容</text>
      </view>

      <!-- 协议内容区域 -->
      <scroll-view scroll-y class="content-scroll" :show-scrollbar="false">
        <view class="content">
          <text class="desc">
            感谢你使用我们的服务。为保障你的权益，请在使用前仔细阅读
            <text class="highlight" @tap="navigateToAgreement('user')">《用户协议》</text>和
            <text class="highlight" @tap="navigateToAgreement('privacy')">《隐私政策》</text>。
          </text>

          <text class="desc">
            我们将严格遵守相关法律法规，采取安全保护措施保护你的个人信息。点击"同意"即表示你已阅读并同意全部条款。
            你可以在【我的-设置-隐私设置】中随时撤回同意或管理你的隐私偏好。
          </text>
        </view>
      </scroll-view>

      <!-- 操作按钮区域 -->
      <view class="actions">
        <button class="btn disagree-btn" @tap="onDisagree" :disabled="loading">暂不使用</button>
        <button class="btn agree-btn" @tap="onAgree" :loading="loading">同意并继续</button>
      </view>
    </view>
  </view>
</template>

<script setup lang="ts">
import { ref } from 'vue'

// 组件属性
const props = defineProps<{
  // 是否显示弹窗
  show: boolean
}>()

// 自定义事件
const emit = defineEmits<{
  // 用户同意
  agree: []
  // 用户拒绝
  disagree: []
  // 点击遮罩（通常不允许关闭）
  'mask-tap': []
}>()

// 本地状态
const loading = ref(false)

// 处理同意
async function onAgree() {
  if (loading.value) return

  loading.value = true
  try {
    // 这里可以添加一些前置逻辑，比如版本检查等

    // 通知父组件用户同意了
    emit('agree')

    // 模拟一点延迟，让用户看到加载状态
    await new Promise((resolve) => setTimeout(resolve, 300))
  } catch (error) {
    console.error('同意操作出错:', error)
    uni.showToast({
      title: '操作失败，请重试',
      icon: 'none',
    })
  } finally {
    loading.value = false
  }
}

// 处理拒绝/暂不使用
function onDisagree() {
  // 先给用户一个确认提示
  uni.showModal({
    title: '提示',
    content: '需要同意用户协议和隐私政策才能使用完整功能。确定要暂不使用吗？',
    confirmText: '确定',
    cancelText: '取消',
    success: (res) => {
      if (res.confirm) {
        // 用户确认拒绝
        emit('disagree')
      }
      // 如果取消，则保持弹窗显示
    },
  })
}

// 处理遮罩点击
function onMaskTap() {
  // 隐私协议弹窗通常不允许点击遮罩关闭
  // 但可以触发一个事件让父组件决定如何处理
  emit('mask-tap')
}

// 跳转到协议页面
function navigateToAgreement(type: string) {
  let url = ''
  switch (type) {
    case 'user':
      url = '/pages/agreement/user-agreement'
      break
    case 'privacy':
      url = '/pages/agreement/privacy-policy'
      break
    default:
      uni.showToast({
        title: '页面开发中',
        icon: 'none',
      })
      return
  }

  uni.navigateTo({
    url,
    fail: () => {
      // 如果页面不存在，显示提示
      uni.showToast({
        title: '页面开发中',
        icon: 'none',
      })
    },
  })
}
</script>

<style scoped>
/* 容器样式 */
.privacy-container {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 9999;
  display: flex;
  align-items: flex-end;
  justify-content: center;
}

/* 遮罩层 */
.mask {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
}

/* 内容卡片 */
.content-card {
  position: relative;
  width: 100%;
  max-width: 750rpx;
  background-color: #ffffff;
  border-radius: 24rpx 24rpx 0 0;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  animation: slideUp 0.3s ease;
}

/* 标题区域 */
.header {
  padding: 40rpx 32rpx 24rpx;
  border-bottom: 1rpx solid #f0f0f0;
  text-align: center;
}

.title {
  display: block;
  font-size: 36rpx;
  font-weight: 600;
  color: #333333;
  margin-bottom: 8rpx;
}

.subtitle {
  display: block;
  font-size: 28rpx;
  color: #666666;
}

/* 滚动内容区域 */
.content-scroll {
  flex: 1;
  min-height: 200rpx;
}

.content {
  padding: 32rpx;
}

/* 描述文本 */
.desc {
  display: block;
  font-size: 28rpx;
  line-height: 1.6;
  color: #333333;
  margin-bottom: 24rpx;
}

.highlight {
  color: #007aff;
}

/* 操作按钮区域 */
.actions {
  display: flex;
  padding: 24rpx 32rpx 96rpx;
  gap: 20rpx;
  border-top: 1rpx solid #f0f0f0;
}

.btn {
  flex: 1;
  height: 88rpx;
  border-radius: 12rpx;
  font-size: 32rpx;
  font-weight: 500;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  margin: 0;
}

.btn::after {
  border: none;
}

/* 拒绝按钮 */
.disagree-btn {
  background-color: #f8f8f8;
  color: #666666;
}

.disagree-btn:active {
  background-color: #eeeeee;
}

/* 同意按钮 */
.agree-btn {
  background-color: #007aff;
  color: #ffffff;
}

.agree-btn:active {
  background-color: #0056cc;
}

/* 动画 */
@keyframes slideUp {
  from {
    transform: translateY(100%);
  }
  to {
    transform: translateY(0);
  }
}

/* 响应式适配 */
@media (max-width: 750rpx) {
  .content-card {
    margin: 0;
  }

  .actions {
    flex-direction: column;
    gap: 16rpx;
  }

  .btn {
    width: 100%;
  }
}
</style>
