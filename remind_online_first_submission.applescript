on run argv
    set candidateMessage to item 1 of argv
    display notification candidateMessage with title "Spaceship Titanic：即将线上测试"
    tell application "Finder"
        activate
        display dialog candidateMessage & return & return & "已完成提交文件与模型重载检查。这是探索性评测，未宣称稳定领先。按钮只关闭提醒，不控制上传。" with title "Spaceship Titanic：已获授权" buttons {"我已看到"} default button "我已看到" with icon note
    end tell
end run
