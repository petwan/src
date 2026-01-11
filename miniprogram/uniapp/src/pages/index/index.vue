<template>
  <view class="index-page">
    <view class="content">
      <text class="title">欢迎使用</text>
      <text class="subtitle">UniApp + Vue3 + TypeScript</text>
    </view>

    <!-- 隐私协议弹窗 -->
    <PrivacyAgreement
      :show="showPrivacyDialog"
      @agree="handlePrivacyAgree"
      @disagree="handlePrivacyDisagree"
    />
  </view>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useAppStore } from '@/store/modules/app'
import { usePrivacyStore } from '@/store/modules/privacy'
import PrivacyAgreement from '@/components/privacy-agreement/privacy-agreement.vue'

// Store
const appStore = useAppStore()
const privacyStore = usePrivacyStore()

// 隐私弹窗显示状态
const showPrivacyDialog = ref(false)

/**
 * 初始化应用
 */
const initApp = () => {
  // 初始化应用状态
  appStore.initApp()

  // 检查隐私政策
  privacyStore.initPrivacy()

  // 如果需要显示隐私弹窗
  if (privacyStore.shouldShowDialog) {
    showPrivacyDialog.value = true
  }
}

/**
 * 处理隐私协议同意
 */
const handlePrivacyAgree = () => {
  showPrivacyDialog.value = false
  privacyStore.agreePrivacyPolicy()

  uni.showToast({
    title: '已同意隐私政策',
    icon: 'success',
  })
}

/**
 * 处理隐私协议拒绝
 */
const handlePrivacyDisagree = () => {
  showPrivacyDialog.value = false
  privacyStore.declinePrivacyPolicy()

  // 用户拒绝后退出应用或跳转到其他页面
  uni.reLaunch({
    url: '/pages/index/index',
  })
}

// 页面加载时初始化
onMounted(() => {
  initApp()
})
</script>

<style lang="scss" scoped>
.index-page {
  width: 100%;
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.content {
  text-align: center;
}

.title {
  display: block;
  font-size: 48rpx;
  font-weight: 600;
  color: #ffffff;
  margin-bottom: 16rpx;
}

.subtitle {
  display: block;
  font-size: 32rpx;
  color: rgba(255, 255, 255, 0.9);
}
</style>
