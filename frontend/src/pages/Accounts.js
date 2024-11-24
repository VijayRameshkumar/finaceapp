const Accounts = () => {
  const url = process.env.REACT_APP_VESSEL_ACCOUNT_URL;
  return (
    <div className="p-4">
      <h2 className="text-2xl font-bold">Accounts</h2>
      <p>Synergy Vessel Accounts</p>

      {/* Embed Power BI report using an iframe */}
      <iframe
        title="Synergy Vessel Accounts - Final"
        width="100%"
        height="500"
        src={url}
        style={{
          border: '1px solid #000', 
          borderRadius: '3px', 
          boxShadow: '0 2px 5px rgba(0, 0, 0, 0.3)'
      }}
      ></iframe>
    </div>
  );
};

export default Accounts;
