<script setup lang="ts">
import { computed } from 'vue'
import { useData } from 'vitepress'

const { frontmatter } = useData()

const isDraft = computed(() => {
  return frontmatter.value.draft === true
})

const showMessage = computed(() => {
  return isDraft.value
})
</script>

<template>
  <div v-if="showMessage" class="content-warning">
    <div class="warning-banner">
      ⚠️ 此页面为草稿版本，内容尚在完善中，可能包含不完整或不准确的信息。
    </div>
  </div>
</template>

<style scoped>
.content-warning {
  margin: 1rem 0;
}

.warning-banner {
  padding: 0.75rem 1rem;
  background-color: var(--vp-c-warning-soft);
  border-left: 4px solid var(--vp-c-warning-1);
  border-radius: 0 4px 4px 0;
  color: var(--vp-c-warning-1);
  font-weight: 500;
}
</style>