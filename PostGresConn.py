import psycopg
import os
from dotenv import load_dotenv

class PostgresSQL:
    def __init__(self):
        # self.type = type
        # self.prompt = SQLPrompt
        load_dotenv()
        self.db_user = os.getenv("DB_USER")
        self.db_port = os.getenv("DB_PORT")
        self.db_password = os.getenv("DB_PASSWORD")
        self.db_name = os.getenv("DB_DBNAME")
        self.db_host = os.getenv("DB_ADDR")

    def conn(self):
        conn = psycopg.connect(
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port
        )
        return conn

    def FetchAllData(self, table):
        conn = self.conn()
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table}")
            columns = [desc[0] for desc in cur.description]  # Extract column names
            all_data = cur.fetchall()
        return [dict(zip(columns, row)) for row in all_data]

    def FetchSpecificData(self, table, condition="", columns="*"):
        """
        Fetch specific data from a table.
        :param table: Table name
        :param condition: Optional SQL condition, e.g. "WHERE id < 5"
        :param columns: Columns to select, default "*"
        """
        conn = self.conn()
        with conn.cursor() as cur:
            cur.execute(f"SELECT {columns} FROM {table} {condition}")
            col_names = [desc[0] for desc in cur.description]
            all_data = cur.fetchall()
        return [dict(zip(col_names, row)) for row in all_data]

    def InsertData(self, table, data: dict):
        """
        Expecting: {"currency": "USD", "forecast": "280K"}
        Returns:
            200 on success,
            400 or other code on failure.
        """
        try:
            conn = self.conn()
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['%s'] * len(data))
            values = tuple(data.values())

            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

            with conn.cursor() as cur:
                cur.execute(query, values)

            conn.commit()
            return 200

        except Exception as e:
            print(f"error as {e}")
            return 400

    def DeleteSpecificData(self, table, data: dict):
        """
        Expecting: {"currency": "USD", "forecast": "280K"}
        Returns:
            200 on success,
            400 or other code on failure.
        """
        try:
            if not data:
                return 400
            conn = self.conn()
            conditions = ' AND '.join([f"{key} = %s" for key in data.keys()])
            values = tuple(data.values())

            query = f"DELETE FROM {table} WHERE {conditions}"

            with conn.cursor() as cur:
                cur.execute(query, values)

            conn.commit()
            return 200

        except Exception as e:
            print(f"error as {e}")
            return 400

    def UpdateSpecificData(self, table, data: dict, conditions: dict):
        """
        Updates rows in the given table.
        
        Args:
            table (str): Table name.
            data (dict): Columns and values to update, e.g. {"forecast": "300K"}.
            conditions (dict): Conditions for the update, e.g. {"currency": "USD"}.

        Returns:
            200 on success,
            400 on failure.
        """
        try:
            if not data or not conditions:
                return 400

            conn = self.conn()

            # Build SET clause
            set_clause = ', '.join([f"{key} = %s" for key in data.keys()])
            set_values = tuple(data.values())

            # Build WHERE clause
            where_clause = ' AND '.join([f"{key} = %s" for key in conditions.keys()])
            where_values = tuple(conditions.values())

            query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"

            with conn.cursor() as cur:
                cur.execute(query, set_values + where_values)

            conn.commit()
            return 200

        except Exception as e:
            print(f"Error in UpdateSpecificData: {e}")
            return 400
