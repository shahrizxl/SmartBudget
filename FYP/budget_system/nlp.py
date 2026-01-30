from nlp_utils import analyze_message

TEST_CASES = [

    # ---- BASIC EXPENSES ----
    "Bought lunch RM12",
    "Paid RM5 for parking",
    "Spent 20 on food",
    "Bought nasi lemak rm6",
    "Tapau chicken rice RM8",
    "Grab RM7",
    "Parking 4",
    "Bought drink 3",

    # ---- MULTI TRANSACTIONS ----
    "Bought lunch RM12 and coffee RM5",
    "Paid RM4 for bus and RM2 for water",
    "RM10 for parking; RM4 toll; RM6 drink",
    "Paid RM10 & RM5 for two items",
    "Bought burger RM10, fries RM5 and drink RM3",
    "Spent rm5, rm3, rm2",
    "Lunch 12 and coffee 5 and snack 3",
    "Grab RM8 and dinner RM15",

    # ---- INCOME ----
    "Received salary RM1200",
    "Got bonus RM300",
    "Parents gave me RM100",
    "Sold old laptop RM500",
    "Won RM50 in a game",
    "Get rm20 cashback",
    "Rebate rm15 from shopee",
    "Bank in rm200 from friend",
    "Duit masuk rm30",

    # ---- MIXED EXPENSE & INCOME ----
    "Received RM100 and spent RM20 on lunch",
    "Got RM30 rebate then bought snacks RM5",
    "Refunded 30 and spent RM10",
    "Sold item RM40 and bought food RM12",

    # ---- MALAYSIAN SLANG ----
    "Tapau nasi goreng rm8",
    "Belanja kawan rm10",
    "Topap rm5 for Hotlink",
    "Beli air rm3",
    "Makan rm12",
    "Minum teh tarik rm2",
    "Grabfood rm18",
    "Foodpanda rm22",
    "Bayar minyak rm20",
    "Bayar tol rm6",

    # ---- VENDORS ----
    "Bought Starbucks RM15",
    "McD RM12",
    "KFC rm18",
    "Shopee rm25",
    "Lazada rm40",
    "Tealive rm8",
    "Secret Recipe rm20",
    "AEON rm35 groceries",
    "FamilyMart rm12",

    # ---- BILLS ----
    "Paid TNB RM60",
    "Paid water bill RM15",
    "Paid wifi RM89",
    "Paid phone bill RM45",
    "Pay Netflix RM17",
    "Spotify RM15",
    "Unifi RM129",
    "Maxis rm80",
    "Insurance rm120",

    # ---- SHOPPING ----
    "Bought clothes rm50",
    "Bought shoes rm120",
    "Shopping rm30",
    "New phone cable rm10",
    "Bought headset rm35",
    "New shirt rm25",
    "Shopee rm18 for screen protector",
    "Shein rm40",

    # ---- TRICKY ----
    "Paid for lunch",
    "Random text without money",
    "Bought something",
    "RM20",
    "20",
    "Lunch rm?",
    "Spent",

    # ---- STRESS CASES ----
    "Grab rm7, lunch 12, rm3 drink, toll rm2, parking rm4 and bought snack 5",
    "Got salary rm1200, spent rm12 breakfast, rm25 groceries, rm7 grab and rm20 dinner",
    "Bought rm4 sweet and rm9 chocolate and rm2 tissue and rm8 grab",
    "Refund rm20, cashback rm5, spent rm14 food, grab rm6, parking 4",
    "Spent rm10 today and rm5 yesterday and got rm20 from friend and bought nasi rm7",
]


def run_tests():
    print("\n======================")
    print(" NLP TEST RESULTS")
    print("======================\n")

    for text in TEST_CASES:
        print(f"INPUT: {text}")
        try:
            result = analyze_message(text)
            print("OUTPUT:", result)
        except Exception as e:
            print("ERROR:", e)
        print("-" * 60)


if __name__ == "__main__":
    run_tests()
