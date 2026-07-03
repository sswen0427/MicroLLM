#include <gtest/gtest.h>

#include "runtime/chat_template.h"

TEST(ChatTemplateTest, FormatsTinyLlamaUserMessage) {
  EXPECT_EQ(runtime::FormatTinyLlamaChatPrompt({
                {.role = runtime::ChatRole::kUser, .content = "Hello"},
            }),
            "<|user|>\nHello</s>\n<|assistant|>");
}

TEST(ChatTemplateTest, FormatsTinyLlamaSystemAndUserMessage) {
  EXPECT_EQ(
      runtime::FormatTinyLlamaChatPrompt({
          {.role = runtime::ChatRole::kSystem, .content = "You are helpful."},
          {.role = runtime::ChatRole::kUser, .content = "Hello"},
      }),
      "<|system|>\nYou are helpful.</s>\n"
      "<|user|>\nHello</s>\n"
      "<|assistant|>");
}

TEST(ChatTemplateTest, FormatsTinyLlamaMultiTurnMessages) {
  EXPECT_EQ(runtime::FormatTinyLlamaChatPrompt({
                {.role = runtime::ChatRole::kUser, .content = "Hello"},
                {.role = runtime::ChatRole::kAssistant, .content = "Hi."},
                {.role = runtime::ChatRole::kUser, .content = "Who are you?"},
            }),
            "<|user|>\nHello</s>\n"
            "<|assistant|>\nHi.</s>\n"
            "<|user|>\nWho are you?</s>\n"
            "<|assistant|>");
}
