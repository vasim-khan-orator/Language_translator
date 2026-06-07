# -----------------------------
# DISPLAY OUTPUT
# -----------------------------
def display_output(
    mode,
    hindi,
    english
):

    print("\n--------------------------------")

    if mode == "LIVE":

        print(f"[LIVE TOKEN]")
        print(f"Hindi   : {hindi}")
        print(f"English : {english}")

    elif mode == "FINAL":

        print(f"[FINAL SENTENCE]")
        print(f"Hindi   : {hindi}")
        print(f"English : {english}")

    print("--------------------------------")