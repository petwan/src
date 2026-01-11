<template>
  <view class="agreement-page">
    <view class="header">
      <text class="title">用户协议</text>
      <text class="date">更新日期：{{ updateDate }}</text>
    </view>

    <scroll-view scroll-y class="content">
      <view class="section">
        <text class="section-title">1. 协议的接受</text>
        <text class="section-text">
          欢迎使用本应用！本用户协议（以下简称"本协议"）是您（用户）与开发者之间关于使用本应用的法律协议。在使用本应用之前，请您仔细阅读本协议。如您不同意本协议的任何内容，请停止使用本应用。
        </text>
      </view>

      <view class="section">
        <text class="section-title">2. 服务内容</text>
        <text class="section-text">
          本应用为用户提供[具体服务内容]。开发者保留随时修改、中断或终止部分或全部服务的权利，而无需对个别用户或第三方承担责任。
        </text>
      </view>

      <view class="section">
        <text class="section-title">3. 用户账号</text>
        <text class="section-text">
          3.1 用户在使用本应用服务时，可能需要注册账号。用户注册时提供的所有资料必须真实、准确、完整。
          </text>
        <text class="section-text">
          3.2 用户应对其账号和密码的安全负全部责任。如发现账号被他人非法使用，应立即通知开发者。
          </text>
        <text class="section-text">
          3.3 用户不得将账号转让、出售或出借给他人使用。因用户保管不善导致的损失，由用户自行承担。
        </text>
      </view>

      <view class="section">
        <text class="section-title">4. 用户行为规范</text>
        <text class="section-text">
          4.1 用户在使用本应用服务时，必须遵守相关法律法规，不得利用本应用从事违法违规活动。
          </text>
        <text class="section-text">
          4.2 禁止用户从事以下行为：
          </text>
        <text class="section-text">
          - 发布或传输任何违法、有害、威胁、辱骂、骚扰、侵犯他人隐私的信息；
          </text>
        <text class="section-text">
          - 侵犯他人知识产权、商业秘密或其他合法权利；
          </text>
        <text class="section-text">
          - 干扰或破坏本应用的服务器或网络；
          </text>
        <text class="section-text">
          - 利用本应用进行任何形式的广告或商业推广；
          </text>
        <text class="section-text">
          - 其他违反本协议或法律法规的行为。
        </text>
      </view>

      <view class="section">
        <text class="section-title">5. 知识产权</text>
        <text class="section-text">
          5.1 本应用中的所有内容，包括但不限于文字、图片、音频、视频、软件、程序、版面设计等，均受知识产权法律法规保护。
          </text>
        <text class="section-text">
          5.2 未经开发者书面许可，用户不得以任何形式使用、复制、修改、传播上述内容。
        </text>
      </view>

      <view class="section">
        <text class="section-title">6. 免责声明</text>
        <text class="section-text">
          6.1 本应用按"现状"提供服务，不保证服务的适用性、无错误或无中断。
          </text>
        <text class="section-text">
          6.2 因不可抗力、网络故障、系统维护等原因导致服务中断或终止，开发者不承担责任。
          </text>
        <text class="section-text">
          6.3 用户因使用本应用而产生的任何直接或间接损失，开发者不承担责任。
        </text>
      </view>

      <view class="section">
        <text class="section-title">7. 协议修改</text>
        <text class="section-text">
          开发者有权根据需要修改本协议，修改后的协议将在应用内公布。如您继续使用本应用，即表示您同意接受修改后的协议。如您不同意修改后的协议，请停止使用本应用。
        </text>
      </view>

      <view class="section">
        <text class="section-title">8. 法律适用</text>
        <text class="section-text">
          本协议的订立、执行、解释及争议解决均适用中华人民共和国法律。如发生争议，双方应友好协商解决；协商不成的，任何一方均可向开发者所在地人民法院提起诉讼。
        </text>
      </view>

      <view class="section">
        <text class="section-title">9. 联系方式</text>
        <text class="section-text">
          如您对本协议有任何疑问或建议，请通过以下方式联系我们：
          </text>
        <text class="section-text">
          邮箱：support@example.com
          </text>
        <text class="section-text">
          客服热线：400-xxx-xxxx
        </text>
      </view>

      <view class="section">
        <text class="section-title">10. 其他</text>
        <text class="section-text">
          本协议自用户开始使用本应用之日起生效。开发者保留对本协议的最终解释权。
        </text>
      </view>
    </scroll-view>
  </view>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const updateDate = ref('2025-01-11')
</script>

<style lang="scss" scoped>
.agreement-page {
  width: 100%;
  min-height: 100vh;
  background-color: #f5f5f5;
  display: flex;
  flex-direction: column;
}

.header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 40rpx 30rpx;
  text-align: center;
}

.title {
  display: block;
  font-size: 40rpx;
  font-weight: 600;
  color: #ffffff;
  margin-bottom: 12rpx;
}

.date {
  display: block;
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.8);
}

.content {
  flex: 1;
  padding: 30rpx;
}

.section {
  background-color: #ffffff;
  border-radius: 16rpx;
  padding: 30rpx;
  margin-bottom: 20rpx;
}

.section-title {
  display: block;
  font-size: 32rpx;
  font-weight: 600;
  color: #333333;
  margin-bottom: 20rpx;
  line-height: 1.5;
}

.section-text {
  display: block;
  font-size: 28rpx;
  color: #666666;
  line-height: 1.8;
  margin-bottom: 16rpx;
  text-align: justify;
}
</style>
