DROP VIEW IF EXISTS all_names;
CREATE VIEW all_names AS
SELECT
	cust_id AS customer_id,
	Name AS name
FROM kyc_with_open_sanctions
UNION
SELECT
	idsender AS customer_id,
	namesender AS name
FROM emt_trxns
UNION
SELECT
	idreceiver AS customer_id,
	namereceiver AS name
FROM emt_trxns
UNION
SELECT
	idsender AS customer_id,
	namesender AS name
FROM wire_trxns
UNION
SELECT
	idreceiver AS customer_id,
	namereceiver AS name
FROM wire_trxns;