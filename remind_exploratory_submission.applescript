on run argv
 set candidateMessage to item 1 of argv
 display notification candidateMessage with title "Spaceship Titanic：即将探索性提交"
 tell application "Finder"
  activate
  display dialog candidateMessage & return & return & "本版本未通过稳定性门槛。本次按你刚才的明确授权，仅提交一次探索性测试。按钮只关闭提醒。" with title "Spaceship Titanic：已授权一次测试" buttons {"我已看到"} default button "我已看到" with icon note
 end tell
end run
