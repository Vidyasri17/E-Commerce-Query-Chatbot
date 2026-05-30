import sqlite3
import random
from faker import Faker

def seed_database():
    # Initialize Faker with fixed seed for reproducibility
    fake = Faker()
    Faker.seed(42)
    random.seed(42)

    db_path = "ecommerce.db"
    print(f"Creating database at {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Enable foreign keys in SQLite
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Drop existing tables if they exist
    cursor.execute("DROP TABLE IF EXISTS returns;")
    cursor.execute("DROP TABLE IF EXISTS orders;")
    cursor.execute("DROP TABLE IF EXISTS products;")
    cursor.execute("DROP TABLE IF EXISTS customers;")

    # Create Tables
    cursor.execute("""
    CREATE TABLE customers (
        customer_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE
    );
    """)

    cursor.execute("""
    CREATE TABLE products (
        product_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL,
        stock INTEGER NOT NULL,
        rating REAL NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE orders (
        order_id TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL,
        product_id TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        price REAL NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('shipped', 'processing', 'delivered', 'cancelled')),
        tracking_info TEXT,
        delivery_date TEXT,
        estimated_delivery_date TEXT,
        order_date TEXT NOT NULL,
        FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
        FOREIGN KEY(product_id) REFERENCES products(product_id) ON DELETE CASCADE
    );
    """)

    cursor.execute("""
    CREATE TABLE returns (
        return_id TEXT PRIMARY KEY,
        order_id TEXT UNIQUE NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('approved', 'pending')),
        refund_amount REAL NOT NULL,
        timeline TEXT NOT NULL,
        FOREIGN KEY(order_id) REFERENCES orders(order_id) ON DELETE CASCADE
    );
    """)

    # --- 1. Populate Customers ---
    customers_data = []
    
    # Specific test customer C1001
    customers_data.append(("C1001", "John Doe", "johndoe@example.com"))
    
    # Generate 29 more customers (total 30)
    for i in range(2, 31):
        cust_id = f"C1{i:03d}"
        name = fake.name()
        email = f"{name.lower().replace(' ', '')}@example.com"
        customers_data.append((cust_id, name, email))

    cursor.executemany("INSERT INTO customers VALUES (?, ?, ?);", customers_data)

    # --- 2. Populate Products ---
    products_data = []
    
    # Pre-defined products to ensure specific test cases
    # P1001 & P1002 are used by John Doe's orders (Leather Jacket and Running Shoes)
    products_data.append(("P1001", "Running Shoes", "Clothing", 89.99, 10, 4.5))
    products_data.append(("P1002", "Leather Jacket", "Clothing", 199.99, 5, 4.8))
    
    # 5 products with ZERO stock
    out_of_stock = [
        ("P1003", "Wireless Headphones", "Electronics", 149.99, 0, 4.6),
        ("P1004", "Air Fryer XL", "Home & Kitchen", 119.99, 0, 4.7),
        ("P1005", "Smartwatch Series 5", "Electronics", 299.99, 0, 4.4),
        ("P1006", "Yoga Mat Pro", "Sports & Outdoors", 45.00, 0, 4.3),
        ("P1007", "Coffee Maker", "Home & Kitchen", 79.99, 0, 4.5)
    ]
    products_data.extend(out_of_stock)

    # General products
    categories = ["Electronics", "Clothing", "Home & Kitchen", "Sports & Outdoors"]
    product_names = {
        "Electronics": ["Smartphone 12", "Bluetooth Speaker", "USB-C Hub", "Tablet Pro", "Laptop Stand"],
        "Clothing": ["Denim Jeans", "Woolen Sweater", "Cotton T-Shirt", "Socks Pack", "Running Shorts"],
        "Home & Kitchen": ["Blender 900W", "Stand Mixer", "Vacuum Cleaner", "Toaster 2-Slice", "Chef Knife Set"],
        "Sports & Outdoors": ["Camping Tent", "Water Bottle 1L", "Backpack 30L", "Sleeping Bag", "Dumbbells Set"]
    }

    # Generate 15 more products (total 22)
    for i in range(8, 23):
        prod_id = f"P1{i:03d}"
        category = random.choice(categories)
        name = random.choice(product_names[category])
        price = round(random.uniform(15.0, 500.0), 2)
        stock = random.randint(5, 50)
        rating = round(random.uniform(3.5, 5.0), 1)
        products_data.append((prod_id, name, category, price, stock, rating))

    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?);", products_data)

    # --- 3. Populate Orders ---
    orders_data = []
    
    # Standard date range for orders
    # We want customer C1001 (John Doe) to have two active orders:
    # O1001: Running Shoes (P1001) - processing
    # O1002: Leather Jacket (P1002) - delivered
    orders_data.append(("O1001", "C1001", "P1001", 1, 89.99, "processing", None, None, "2026-06-02", "2026-05-28"))
    orders_data.append(("O1002", "C1001", "P1002", 1, 199.99, "delivered", "TRK123456789", "2026-05-20", None, "2026-05-15"))

    # Need at least 10 customers with multiple orders
    # Let's assign multiple orders to customers C1002 through C1011 (10 customers)
    statuses = ["delivered", "shipped", "processing", "cancelled"]
    
    # We seed these systematically
    order_id_counter = 1003
    for i in range(2, 12):
        cust_id = f"C1{i:03d}"
        
        # Customer will have 2 or 3 orders
        num_orders = random.randint(2, 3)
        for _ in range(num_orders):
            ord_id = f"O{order_id_counter}"
            order_id_counter += 1
            
            # Select random product (ensure we don't pick zero stock for non-delivered/non-cancelled orders, or just pick any)
            prod = random.choice(products_data)
            prod_id = prod[0]
            price = prod[3]
            
            status = random.choice(statuses)
            qty = random.randint(1, 2)
            
            order_date = fake.date_between(start_date="-60d", end_date="-5d").strftime("%Y-%m-%d")
            
            if status == "delivered":
                delivery_date = fake.date_between(start_date="-4d", end_date="today").strftime("%Y-%m-%d")
                est_date = None
                tracking = f"TRK{random.randint(100000000, 999999999)}"
            elif status == "shipped":
                delivery_date = None
                est_date = fake.date_between(start_date="today", end_date="+5d").strftime("%Y-%m-%d")
                tracking = f"TRK{random.randint(100000000, 999999999)}"
            elif status == "processing":
                delivery_date = None
                est_date = fake.date_between(start_date="+3d", end_date="+10d").strftime("%Y-%m-%d")
                tracking = None
            else: # cancelled
                delivery_date = None
                est_date = None
                tracking = None
                
            orders_data.append((ord_id, cust_id, prod_id, qty, price, status, tracking, delivery_date, est_date, order_date))

    # Add single orders for some other customers to reach ~50 orders
    for i in range(12, 31):
        cust_id = f"C1{i:03d}"
        ord_id = f"O{order_id_counter}"
        order_id_counter += 1
        
        prod = random.choice(products_data)
        prod_id = prod[0]
        price = prod[3]
        status = random.choice(statuses)
        qty = random.randint(1, 2)
        
        order_date = fake.date_between(start_date="-30d", end_date="-2d").strftime("%Y-%m-%d")
        
        if status == "delivered":
            delivery_date = fake.date_between(start_date="-2d", end_date="today").strftime("%Y-%m-%d")
            est_date = None
            tracking = f"TRK{random.randint(100000000, 999999999)}"
        elif status == "shipped":
            delivery_date = None
            est_date = fake.date_between(start_date="today", end_date="+5d").strftime("%Y-%m-%d")
            tracking = f"TRK{random.randint(100000000, 999999999)}"
        elif status == "processing":
            delivery_date = None
            est_date = fake.date_between(start_date="+3d", end_date="+7d").strftime("%Y-%m-%d")
            tracking = None
        else: # cancelled
            delivery_date = None
            est_date = None
            tracking = None
            
        orders_data.append((ord_id, cust_id, prod_id, qty, price, status, tracking, delivery_date, est_date, order_date))

    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", orders_data)

    # --- 4. Populate Returns ---
    # Enforce referential integrity: returns must be linked to DELIVERED orders
    delivered_orders = [o for o in orders_data if o[5] == "delivered" and o[0] != "O1002"] # skip John Doe's O1002 so it remains returnable in the test conversation
    
    returns_data = []
    return_statuses = ["approved", "pending"]
    
    # We want at least 5 returns to test conditions
    for i in range(min(len(delivered_orders), 8)):
        ord_id = delivered_orders[i][0]
        status = random.choice(return_statuses)
        
        # refund amount is qty * price
        qty = delivered_orders[i][3]
        price = delivered_orders[i][4]
        refund = round(qty * price, 2)
        
        ret_id = f"R100{i+1}"
        timeline = "3-5 business days" if status == "approved" else "Under review (1-2 business days)"
        returns_data.append((ret_id, ord_id, status, refund, timeline))

    cursor.executemany("INSERT INTO returns VALUES (?, ?, ?, ?, ?);", returns_data)

    # Commit and finalize
    conn.commit()

    # Print summary row counts for verification
    print("Database seeding completed successfully.")
    print("--- Row Count Summary ---")
    for table in ["customers", "products", "orders", "returns"]:
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        print(f"Table '{table}': {count} rows")

    # Verify constraint requirements
    # 1. At least 10 customers have multiple orders
    cursor.execute("""
    SELECT customer_id, COUNT(*) 
    FROM orders 
    GROUP BY customer_id 
    HAVING COUNT(*) > 1;
    """)
    multi_order_custs = len(cursor.fetchall())
    print(f"Customers with multiple orders: {multi_order_custs} (Required: >= 10)")

    # 2. At least 5 products have zero stock
    cursor.execute("SELECT COUNT(*) FROM products WHERE stock = 0;")
    zero_stock_prods = cursor.fetchone()[0]
    print(f"Products with zero stock: {zero_stock_prods} (Required: >= 5)")

    conn.close()

if __name__ == "__main__":
    seed_database()
