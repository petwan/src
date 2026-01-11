<script setup lang="ts">
import { onLaunch, onShow, onHide } from "@dcloudio/uni-app"

// 初始化更新检查（useAppUpdate 已通过 unplugin-auto-import 自动导入）
const { checkUpdate } = useAppUpdate({
  autoCheck: true, // 自动检查更新
  title: "更新提示",
  readyContent: "新版本已经准备好，是否重启应用？",
  failContent: "新版本下载失败，请检查网络后重试",
})

onLaunch(() => {
  console.log("App Launch")

  // 检查软件更新(通过 hooks 自动执行)
  checkUpdate()
})

onShow(() => {
  console.log("App Show")
})

onHide(() => {
  console.log("App Hide")
})
</script>

<style lang="scss">
/* stylelint-disable selector-type-no-unknown */
button::after {
  border: none;
}

swiper,
scroll-view {
  flex: 1;
  height: 100%;
  overflow: hidden;
}

image {
  width: 100%;
  height: 100%;
  vertical-align: middle;
}

// 单行省略，优先使用 unocss: text-ellipsis
.ellipsis {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

// 两行省略
.ellipsis-2 {
  display: -webkit-box;
  overflow: hidden;
  text-overflow: ellipsis;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

// 三行省略
.ellipsis-3 {
  display: -webkit-box;
  overflow: hidden;
  text-overflow: ellipsis;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
}

map .csssprite {
  display: none !important;
}
</style>
