-- Launch asynchronously before an already-authorized, qualified upload.
-- Dismissing this reminder is not an authorization or submission action.
on run argv
    set candidateMessage to item 1 of argv
    display notification candidateMessage with title "Spaceship Titanic：即将自动提交"
    tell application "Finder"
        activate
        display dialog candidateMessage & return & return & "此候选已通过本轮稳定性审核，将按你已有的授权自动提交 Kaggle。按钮只关闭提醒，不控制提交。" with title "Spaceship Titanic：已授权线上测试" buttons {"我已看到"} default button "我已看到" with icon note
    end tell
end run
