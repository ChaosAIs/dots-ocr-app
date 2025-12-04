import { AgenticChatBot } from "../components/chatbot/AgenticChatBot";
import "./chat.scss";

export const Chat = () => {
  return (
    <div className="chat-page-container">
      <div className="chat-page-content">
        <AgenticChatBot />
      </div>
    </div>
  );
};

