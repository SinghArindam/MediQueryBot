/* MediQueryBot – front-end
   Markdown answers + latency display
*/
const API = {
  newChat : "/chats",
  list    : "/chats",
  history : id => `/chats/${id}/history`,
  ask     : id => `/chats/${id}/ask`
};

let currentId = null;

/* DOM refs ---------------------------------------------------- */
const chatList = document.getElementById("chat-list");
const newBtn   = document.getElementById("new-chat-btn");
const messages = document.getElementById("messages");
const input    = document.getElementById("user-input");
const sendBtn  = document.getElementById("send-btn");
const form     = document.getElementById("chat-form");

/* helpers ----------------------------------------------------- */
function bubble(html, who="user"){
  const div=document.createElement("div");
  div.className=who==="user"
      ?"self-end bg-blue-600 text-white p-3 max-w-[75%] rounded-2xl rounded-br-none shadow"
      :"self-start bg-green-600 text-white p-3 max-w-[75%] rounded-2xl rounded-bl-none shadow";
  div.innerHTML=html;
  messages.appendChild(div);
  messages.scrollTop=messages.scrollHeight;
}

function renderAssistant(raw,lat){
  const thinkRE=/<think>([\s\S]*?)<\/think>/i;
  let think="", ans=raw;
  const m=raw.match(thinkRE);
  if(m){ think=m[1].trim(); ans=raw.replace(thinkRE,"").trim(); }
  let html=marked.parse(ans);
  if(think){
    html+=`<details class="mt-3 bg-black/30 p-2 rounded">
             <summary class="cursor-pointer text-gold">💭 Reasoning</summary>
             <pre class="whitespace-pre-wrap text-xs mt-2">${think}</pre>
           </details>`;
  }
  html+=`<div class="text-xs text-white/70 mt-2">Responded in ${lat}s</div>`;
  return html;
}

/* sidebar ----------------------------------------------------- */
async function refreshChatList(sel){
  const list=await fetch(API.list).then(r=>r.json());
  chatList.innerHTML="";
  list.forEach(c=>{
    const li=document.createElement("li");
    li.textContent=c.title||c.id;
    li.className=(c.id===sel?"bg-gold text-black":"")+
                 " block px-3 py-1 rounded cursor-pointer hover:bg-white/10";
    li.onclick=()=>loadChat(c.id);
    chatList.appendChild(li);
  });
}

/* history ----------------------------------------------------- */
async function loadChat(id){
  currentId=id; messages.innerHTML="";
  await refreshChatList(id);
  const hist=await fetch(API.history(id)).then(r=>r.json());
  hist.forEach(t=>{
    if(t.role==="assistant"){
      bubble(renderAssistant(t.content,t.latency??"--"),"bot");
    }else{
      bubble(marked.parseInline(t.content),"user");
    }
  });
}

/* new chat ---------------------------------------------------- */
async function createChat(){
  const {id}=await fetch(API.newChat,{method:"POST"}).then(r=>r.json());
  await loadChat(id);
}

/* send -------------------------------------------------------- */
async function askLLM(q){
  if(!currentId) await createChat();
  bubble(marked.parseInline(q),"user");
  input.value="";
  const typing=document.createElement("div");
  typing.className="self-start italic text-white/70";
  typing.textContent="MediQueryBot is typing…";
  messages.appendChild(typing); messages.scrollTop=messages.scrollHeight;

  try{
    const res=await fetch(API.ask(currentId),{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({message:q})
    });
    if(!res.ok) throw new Error(await res.text());
    const {reply,latency}=await res.json();
    messages.removeChild(typing);
    bubble(renderAssistant(reply,latency),"bot");
  }catch(e){
    messages.removeChild(typing);
    bubble(`<span class="text-red-400">⚠️ ${e.message}</span>`,"bot");
  }
}

/* events ------------------------------------------------------ */
newBtn.onclick=createChat;
sendBtn.onclick=()=>{const t=input.value.trim();if(t)askLLM(t);};
form.addEventListener("submit",e=>{e.preventDefault();const t=input.value.trim();if(t)askLLM(t);});

/* boot -------------------------------------------------------- */
refreshChatList();
