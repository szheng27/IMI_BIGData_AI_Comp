DROP VIEW IF EXISTS kyc_features;
CREATE VIEW kyc_features AS
SELECT
	Name,
	Gender,
	Occupation,
	Age,
	Tenure,
	cust_id,
	no_wire_trxns_sent,
	wire_total_amnt_sent,
	no_international_wire_trxns_sent,
	no_wire_trxns_received,
	wire_total_amnt_received,
	no_international_wire_trxns_received,
	no_emt_trxns_sent,
	emt_total_amnt_sent,
	no_emt_trxns_received,
	emt_total_amnt_received,
	no_cash_deposits,
	total_cash_deposit,
	no_cash_withdraws,
	total_cash_withdraw,
	(no_international_wire_trxns_sent + no_international_wire_trxns_received) AS total_international_trxns,
	(no_wire_trxns_sent + no_wire_trxns_received) AS total_no_wire_trxns,
	(no_emt_trxns_sent + no_emt_trxns_received) AS total_no_emt_trxns,
	(no_cash_deposits + no_cash_withdraws) AS total_no_cash_trxns,
	(wire_total_amnt_received - wire_total_amnt_sent) AS wire_trxn_balance,
	(emt_total_amnt_received - emt_total_amnt_sent) AS emt_trxn_balance,
	(total_cash_deposit - total_cash_withdraw) AS cash_balance,
	label
FROM
(
SELECT
    k.Name,
    k.Gender,
    k.Occupation,
    k.Age,
    k.Tenure,
    k.cust_id,
    IFNULL(w.no_wire_trxns_sent,0) AS no_wire_trxns_sent,
    IFNULL(w.wire_total_amnt_sent,0) AS wire_total_amnt_sent,
    IFNULL(w.no_international_wire_trxns_sent,0) AS no_international_wire_trxns_sent,
    IFNULL(wr.no_wire_trxns_received,0) AS no_wire_trxns_received,
    IFNULL(wr.wire_total_amnt_received,0) AS wire_total_amnt_received,
    IFNULL(wr.no_international_wire_trxns_received,0) AS no_international_wire_trxns_received,
    IFNULL(e.no_emt_trxns_sent,0) AS no_emt_trxns_sent,
    IFNULL(e.emt_total_amnt_sent,0) AS emt_total_amnt_sent,
    IFNULL(er.no_emt_trxns_received,0) AS no_emt_trxns_received,
    IFNULL(er.emt_total_amnt_received,0) AS emt_total_amnt_received,
    IFNULL(c.no_cash_deposits,0) AS no_cash_deposits,
    IFNULL(c.total_cash_deposit,0) AS total_cash_deposit,
    IFNULL(cw.no_cash_withdraws,0) AS no_cash_withdraws,
    IFNULL(cw.total_cash_withdraw,0) AS total_cash_withdraw,
    k.label
FROM kyc k
LEFT JOIN (
    SELECT
        id_sender AS cust_id,
        COUNT(wire_trxn_id) AS no_wire_trxns_sent,
        SUM(wire_value) AS wire_total_amnt_sent,
        SUM(CASE WHEN country_receiver <> country_sender THEN 1 ELSE 0 END) AS no_international_wire_trxns_sent
    FROM wire_trxns
    GROUP BY id_sender
) w ON k.cust_id = w.cust_id
LEFT JOIN (
    SELECT
        id_receiver AS cust_id,
        COUNT(wire_trxn_id) AS no_wire_trxns_received,
        SUM(wire_value) AS wire_total_amnt_received,
        SUM(CASE WHEN country_receiver <> country_sender THEN 1 ELSE 0 END) AS no_international_wire_trxns_received
    FROM wire_trxns
    GROUP BY id_receiver
) wr ON k.cust_id = wr.cust_id
LEFT JOIN (
    SELECT
        id_sender AS cust_id,
        COUNT(emt_trxn_id) AS no_emt_trxns_sent,
        SUM(emt_value) AS emt_total_amnt_sent
    FROM emt_trxns
    GROUP BY id_sender
) e ON k.cust_id = e.cust_id
LEFT JOIN (
    SELECT
        id_receiver AS cust_id,
        COUNT(emt_trxn_id) AS no_emt_trxns_received,
        SUM(emt_value) AS emt_total_amnt_received
    FROM emt_trxns
    GROUP BY id_receiver
) er ON k.cust_id = er.cust_id
LEFT JOIN (
    SELECT
        cust_id,
        COUNT(cash_trxn_id) AS no_cash_deposits,
        SUM(amount) AS total_cash_deposit
    FROM cash_trxns
    WHERE type = 'deposit'
    GROUP BY cust_id
) c ON k.cust_id = c.cust_id
LEFT JOIN (
    SELECT
        cust_id,
        COUNT(cash_trxn_id) AS no_cash_withdraws,
        SUM(amount) AS total_cash_withdraw
    FROM cash_trxns
    WHERE type = 'withdrawal'
    GROUP BY cust_id
) cw ON k.cust_id = cw.cust_id
);