import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ChatServiceService {

  private apiUrl = 'http://localhost:8000/generate'; // Node.js backend URL

  constructor(private http: HttpClient) {}

  sendMessage(prompt: string): Observable<any> {
      return this.http.post(this.apiUrl, { prompt });
  }

}
