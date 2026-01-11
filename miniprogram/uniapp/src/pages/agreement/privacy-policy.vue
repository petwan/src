<template>
  <view class="agreement-page">
    <view class="header">
      <text class="title">隐私政策</text>
      <text class="date">更新日期：{{ updateDate }}</text>
    </view>

    <scroll-view scroll-y class="content">
      <view class="section">
        <text class="section-title">引言</text>
        <text class="section-text">
          我们深知个人信息对您的重要性，并会尽全力保护您的个人信息安全可靠。我们致力于维持您对我们的信任，恪守以下原则，保护您的个人信息：权责一致原则、目的明确原则、选择同意原则、最少够用原则、确保安全原则、主体参与原则、公开透明原则等。
        </text>
        <text class="section-text">
          本隐私政策适用于您通过本应用产品或服务访问和使用我们的产品和服务。
        </text>
      </view>

      <view class="section">
        <text class="section-title">一、我们如何收集和使用您的个人信息</text>
        <text class="section-text">
          1.1 当您使用本应用时，我们会收集以下信息：
          </text>
        <text class="section-text">
          • 设备信息：包括设备型号、操作系统版本、唯一设备标识符等；
          </text>
        <text class="section-text">
          • 网络信息：包括IP地址、网络类型、网络运营商等；
          </text>
        <text class="section-text">
          • 日志信息：包括应用使用记录、操作日志、错误日志等；
          </text>
        <text class="section-text">
          • 位置信息：在您授权的情况下，收集您的地理位置信息；
          </text>
        <text class="section-text">
          • 账户信息：如果您注册账户，我们会收集您提供的用户名、密码、手机号、邮箱等信息。
        </text>
        <text class="section-text">
          1.2 我们使用您的个人信息用于以下用途：
          </text>
        <text class="section-text">
          • 提供和维护本应用的服务功能；
          </text>
        <text class="section-text">
          • 改善和优化用户体验；
          </text>
        <text class="section-text">
          • 防范和打击欺诈、滥用等行为；
          </text>
        <text class="section-text">
          • 进行产品分析和数据分析；
          </text>
        <text class="section-text">
          • 向您发送服务通知、活动通知等信息（需您同意）；
          </text>
        <text class="section-text">
          • 法律法规规定的其他用途。
        </text>
      </view>

      <view class="section">
        <text class="section-title">二、我们如何共享、转让、公开披露您的个人信息</text>
        <text class="section-text">
          2.1 共享
          </text>
        <text class="section-text">
          除以下情况外，我们不会与任何公司、组织和个人共享您的个人信息：
          </text>
        <text class="section-text">
          • 在获取明确同意的情况下共享；
          </text>
        <text class="section-text">
          • 根据法律法规或政府主管部门要求；
          </text>
        <text class="section-text">
          • 与我们的关联公司共享，但我们只会共享必要的个人信息；
          </text>
        <text class="section-text">
          • 与授权合作伙伴共享，但我们仅会共享必要的信息，且要求合作伙伴不得将您的个人信息用于任何其他目的。
        </text>
        <text class="section-text">
          2.2 转让
          </text>
        <text class="section-text">
          我们不会将您的个人信息转让给任何公司、组织和个人，但以下情况除外：
          </text>
        <text class="section-text">
          • 获取您的明确同意；
          </text>
        <text class="section-text">
          • 根据法律法规或司法裁判要求；
          </text>
        <text class="section-text">
          • 在涉及合并、收购或资产转让时。
        </text>
        <text class="section-text">
          2.3 公开披露
          </text>
        <text class="section-text">
          我们仅会在以下情况下公开披露您的个人信息：
          </text>
        <text class="section-text">
          • 获得您的明确同意；
          </text>
        <text class="section-text">
          • 根据法律法规或政府主管部门要求。
        </text>
      </view>

      <view class="section">
        <text class="section-title">三、我们如何保护您的个人信息</text>
        <text class="section-text">
          3.1 我们已采取符合业界标准的安全防护措施来保护您的个人信息安全，包括但不限于：
          </text>
        <text class="section-text">
          • 采用加密技术对个人信息进行加密存储；
          </text>
        <text class="section-text">
          • 采用严格的数据访问控制措施，限制对个人信息的访问权限；
          </text>
        <text class="section-text">
          • 建立数据安全管理制度和应急预案，定期进行安全评估。
        </text>
        <text class="section-text">
          3.2 如果发生个人信息泄露事件，我们将按照法律法规的要求及时告知您，并采取补救措施。
        </text>
        <text class="section-text">
          3.3 尽管我们采取了安全措施，但请注意，没有任何安全措施是绝对安全的。请您妥善保管个人信息，不要将密码等敏感信息告知他人。
        </text>
      </view>

      <view class="section">
        <text class="section-title">四、您的权利</text>
        <text class="section-text">
          按照中国相关的法律法规，您对您的个人信息享有以下权利：
          </text>
        <text class="section-text">
          • 访问您的个人信息；
          </text>
        <text class="section-text">
          • 更正您的个人信息；
          </text>
        <text class="section-text">
          • 删除您的个人信息；
          </text>
        <text class="section-text">
          • 改变您授权同意的范围；
          </text>
        <text class="section-text">
          • 注销您的账户；
          </text>
        <text class="section-text">
          • 获取个人信息副本。
        </text>
        <text class="section-text">
          如您需要行使上述权利，请通过本隐私政策公布的联系方式与我们联系。
        </text>
      </view>

      <view class="section">
        <text class="section-title">五、我们如何处理未成年人的个人信息</text>
        <text class="section-text">
          如果您是未成年人，请在监护人的指导下使用本应用。如果您是未成年人的监护人，请您对未成年人使用本应用的行为进行监护。如发现未成年人未经监护人同意提供了个人信息，请通过本隐私政策公布的联系方式与我们联系，我们会及时删除相关信息。
        </text>
      </view>

      <view class="section">
        <text class="section-title">六、您的个人信息如何在全球范围转移</text>
        <text class="section-text">
          我们在中华人民共和国境内收集和产生的个人信息，将存储在中华人民共和国境内。如需跨境传输，我们将按照法律法规的要求进行安全评估，并确保您的个人信息得到充分的保护。
        </text>
      </view>

      <view class="section">
        <text class="section-title">七、本隐私政策的更新</text>
        <text class="section-text">
          我们可能适时更新本隐私政策的条款，更新后的政策将在应用内公布。如您继续使用本应用，即表示您同意接受更新后的隐私政策。
        </text>
      </view>

      <view class="section">
        <text class="section-title">八、如何联系我们</text>
        <text class="section-text">
          如果您对本隐私政策有任何疑问、意见或建议，或需要行使您的个人信息权利，请通过以下方式与我们联系：
          </text>
        <text class="section-text">
          邮箱：privacy@example.com
          </text>
        <text class="section-text">
          客服热线：400-xxx-xxxx
          </text>
        <text class="section-text">
          通讯地址：[详细地址]
        </text>
        <text class="section-text">
          我们将在收到您的反馈后尽快回复您的请求。
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
