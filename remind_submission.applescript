on run argv
    set candidateMessage to item 1 of argv
    tell application "Finder"
        activate
        display dialog candidateMessage & return & return & "请回到聊天确认是否提交 Kaggle。此弹窗只作提醒，点击按钮不会提交。" with title "Spaceship Titanic：候选待确认" buttons {"我已看到"} default button "我已看到" with icon note
    end tell
end run
