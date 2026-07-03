#include "runtime/chat_template.h"

#include <string>
#include <vector>

namespace runtime {
namespace {

constexpr char kUserRole[] = "<|user|>\n";
constexpr char kSystemRole[] = "<|system|>\n";
constexpr char kAssistantRole[] = "<|assistant|>";
constexpr char kEosToken[] = "</s>\n";

}  // namespace

std::string FormatTinyLlamaChatPrompt(
    const std::vector<ChatMessage>& messages) {
  std::string formatted;
  for (const ChatMessage& message : messages) {
    if (message.content.empty()) {
      continue;
    }
    switch (message.role) {
      case ChatRole::kSystem:
        formatted.append(kSystemRole);
        break;
      case ChatRole::kUser:
        formatted.append(kUserRole);
        break;
      case ChatRole::kAssistant:
        formatted.append(kAssistantRole);
        formatted.push_back('\n');
        break;
    }
    formatted.append(message.content);
    formatted.append(kEosToken);
  }
  formatted.append(kAssistantRole);
  return formatted;
}

}  // namespace runtime
