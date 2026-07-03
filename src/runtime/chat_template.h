#pragma once

#include <string>
#include <vector>

namespace runtime {

enum class ChatRole {
  kSystem,
  kUser,
  kAssistant,
};

struct ChatMessage {
  ChatRole role = ChatRole::kUser;
  std::string content;
};

std::string FormatTinyLlamaChatPrompt(
    const std::vector<ChatMessage>& messages);

}  // namespace runtime
