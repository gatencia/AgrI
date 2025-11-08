import modal

app = modal.App.lookup("my-app", create_if_missing=True)

sb = modal.Sandbox.create(app=app)

p = sb.exec("python", "-c", "print('hello')", timeout=3)
print(p.stdout.read())

p = sb.exec("bash", "-c", "for i in {1..10}; do date +%T; sleep 0.5; done", timeout=5)
for line in p.stdout:
    # Avoid double newlines by using end="".
    print(line, end="")

sb.terminate()