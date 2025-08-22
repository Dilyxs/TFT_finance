import asyncio
from datetime import datetime
from MetaApiConn import MetaV2
from PostGresConn import PostgresSQL

async def DeployAccounts(db, Deployement=True):
    all_cc = db.FetchAllData('accountdata')  # This is sync, we keep it for now

    for acc in all_cc:
        account_id = acc.get('account_id', None)
        access_token = acc.get('access_token', None)
        meta = MetaV2(access_token, account_id)
        if Deployement:
            await meta.deploy_account()
        else:
            await meta.undeploy_account()

async def main():
    db = PostgresSQL()
    now = datetime.utcnow()
    if 40< now.minute < 59:
        await DeployAccounts(db, Deployement=True)
    elif 0< now.minute <20: 
        await DeployAccounts(db, Deployement=False)

# Run the asyncio event loop
asyncio.run(main())

