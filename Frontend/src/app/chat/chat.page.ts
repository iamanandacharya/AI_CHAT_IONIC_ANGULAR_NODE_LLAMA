import { Component, OnInit } from '@angular/core';
import { ChatServiceService } from '../services/chat-service.service'
 
@Component({
  selector: 'app-chat',
  templateUrl: './chat.page.html',
  styleUrls: ['./chat.page.scss'],
  standalone: false,
})
export class ChatPage implements OnInit {

  messages = [{ sender: '', text: '' }];
  userMessage = '';

  constructor(private chatService: ChatServiceService) {}

  ngOnInit(): void {
    
  }
  sendMessage() {
      const userText = this.userMessage.trim();
      if (!userText) return;

      this.messages.push({ sender: 'user', text: userText });
      this.userMessage = '';

      this.chatService.sendMessage(userText).subscribe((response) => {
          this.messages.push({ sender: 'bot', text: response.response });
      });
  }
}
