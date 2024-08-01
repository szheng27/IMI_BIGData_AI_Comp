DROP VIEW IF EXISTS transaction_network;
CREATE VIEW transaction_network AS
SELECT
	idsender,
	idreceiver,
	namesender,
	namereceiver,
	wirevalue AS transfer_amount,
	1 AS transfer_type
FROM wire_trxns
UNION ALL
SELECT
	idsender,
	idreceiver,
	namesender,
	namereceiver,
	emtvalue AS transfer_amount,
	0 AS transfer_type
FROM emt_trxns;