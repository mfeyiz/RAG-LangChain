const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');

// API endpoint'i - RAG sistemi için /ask kullanıyoruz
const API_URL = 'http://localhost:5000/ask';

// Mesaj gönderme fonksiyonu
async function sendMessage() {
    const message = userInput.value.trim();
    
    if (!message) return;
    
    addMessage(message, 'user');
    userInput.value = '';
    autoResize();
    
    sendButton.disabled = true;
    
    const typingIndicator = addTypingIndicator();
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: message })
        });
        
        if (!response.ok) {
            throw new Error('API isteği başarısız oldu');
        }
        
        typingIndicator.remove();
        
        const botMessageDiv = createBotMessage();
        const contentDiv = botMessageDiv.querySelector('.message-content');
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        let searchResults = null;
        let buffer = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value, { stream: true });
            buffer += chunk;
            const lines = buffer.split('\n');
            buffer = lines.pop();
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const content = line.slice(6);
                    
                    if (content.startsWith('__SEARCH_RESULTS__')) {
                        const jsonStr = content.match(/__SEARCH_RESULTS__(.+)__END_SEARCH__/)[1];
                        searchResults = JSON.parse(jsonStr);
                        showSearchResults(searchResults);
                    } else {
                        fullText += content;
                        contentDiv.textContent = fullText;
                        scrollToBottom();
                    }
                }
            }
        }
        
    } catch (error) {
        console.error('Hata:', error);
        typingIndicator.remove();
        addMessage('Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.', 'bot');
    } finally {
        sendButton.disabled = false;
        userInput.focus();
    }
}

// Mesaj ekleme fonksiyonu
function addMessage(text, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = type === 'user' ? '👤' : '🤖';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.textContent = text;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    return messageDiv;
}

// Bot mesajı oluşturma fonksiyonu (streaming için)
function createBotMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    return messageDiv;
}

// Yazıyor göstergesi ekleme
function addTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    messageDiv.id = 'typingIndicator';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';
    
    const content = document.createElement('div');
    content.className = 'message-content typing-indicator';
    content.innerHTML = '<span></span><span></span><span></span>';
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    return messageDiv;
}

// Arama sonuçlarını göster
function showSearchResults(results) {
    const resultsDiv = document.createElement('div');
    resultsDiv.className = 'search-results';
    
    const header = document.createElement('div');
    header.className = 'search-results-header';
    header.innerHTML = `<strong>📚 Bulunan Dokümanlar (${results.length})</strong>`;
    resultsDiv.appendChild(header);
    
    results.forEach((result, index) => {
        const docDiv = document.createElement('div');
        docDiv.className = `search-result-item ${result.relevant ? 'relevant' : 'not-relevant'}`;
        
        const docHeader = document.createElement('div');
        docHeader.className = 'search-result-header';
        docHeader.innerHTML = `
            <span class="doc-number">Doküman ${index + 1}</span>
            <span class="doc-score">Skor: ${result.score.toFixed(4)}</span>
            <span class="doc-status">${result.relevant ? '✓ İlgili' : '✗ İlgisiz'}</span>
        `;
        
        const docContent = document.createElement('div');
        docContent.className = 'search-result-content';
        const isLong = result.content.length > 500;
        docContent.textContent = result.content.substring(0, 500) + (isLong ? '...' : '');
        
        docDiv.appendChild(docHeader);
        docDiv.appendChild(docContent);
        
        if (isLong) {
            const showMoreBtn = document.createElement('button');
            showMoreBtn.className = 'show-more-btn';
            showMoreBtn.textContent = 'Tümünü Göster';
            showMoreBtn.onclick = () => {
                if (docContent.textContent.endsWith('...')) {
                    docContent.textContent = result.content;
                    showMoreBtn.textContent = 'Daha Az Göster';
                } else {
                    docContent.textContent = result.content.substring(0, 500) + '...';
                    showMoreBtn.textContent = 'Tümünü Göster';
                }
            };
            docDiv.appendChild(showMoreBtn);
        }
        
        resultsDiv.appendChild(docDiv);
    });
    
    chatMessages.appendChild(resultsDiv);
    scrollToBottom();
}

// Scroll fonksiyonu
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Textarea auto-resize
function autoResize() {
    userInput.style.height = 'auto';
    userInput.style.height = userInput.scrollHeight + 'px';
}

// Event listeners
sendButton.addEventListener('click', sendMessage);

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

userInput.addEventListener('input', autoResize);

// Sayfa yüklendiğinde scroll'u en alta getir
window.addEventListener('load', scrollToBottom);
