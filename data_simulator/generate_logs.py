import random
downtime_log_templates = {
    'TubingBlockage': {
        'MaintenanceNotes': [
            "Scheduled hot oil circulation due to backpressure buildup.",
            "Injected wax inhibitor following signs of progressive tubing restriction.",
            "Initiated tubing flush; flow decline observed over multiple shifts.",
            "Corrective sweep with scale dissolver programmed during next cycle.",
            "Flagged progressive blockage; isolating affected zone for inspection.",
            "Scheduled temperature-assisted treatment to mitigate suspected paraffin accumulation."
        ],
        'Observations': [
            "Backpressure increased slowly over ~36h, flow decline measured at 14%. Blockage suspected mid-string based on differential build-up.",
            "Typical wax signature observed—flow decrease not accompanied by vibration or temp spike. Classic low-rate buildup scenario.",
            "No abrupt behavior. Downstream pressure stable while tubing pressure trends upward—could be scale formation along the inner walls.",
            "Inspection confirmed mild deposit buildup between joints 3 and 5. Symptoms match prior wax deposition event timeline.",
            "Pressure vs. flow curve shows narrowing characteristic of internal restriction. Flow resistance profile deviating from standard.",
            "Cooling differential and viscosity profile indicate paraffin deposition rather than inorganic scale."
        ]
    },
    'ChokeErosion': {
        'MaintenanceNotes': [
            "Choke replacement requested following confirmed orifice wear.",
            "Flow instability led to visual inspection—erosion marks identified.",
            "Surging rates attributed to progressive choke degradation.",
            "Preventative choke maintenance rescheduled ahead of timeline.",
            "Line vibration near choke prompted early shutdown for swap.",
            "Choke body showed early signs of wear—bypass temporarily enabled."
        ],
        'Observations': [
            "Output flowrate variance increased steadily over past 24h. Trend consistent with prior choke erosion incidents.",
            "High-pitch valve noise correlates with flow instability. Likely micro-cracking around choke orifice.",
            "Flow surges match frequency signature of internal metal loss. Equipment runtime: 630 hours—exceeds typical erosion limit.",
            "Delta-P spike observed at choke inlet suggests edge deformation. Partial restriction forming downstream.",
            "Post-op inspection revealed grooving on the choke wall—caused by high-velocity sand impact.",
            "Rate trend shows sharp but periodic recoveries—symptom of fluctuating orifice shape due to erosion."
        ]
    },
    'LiquidLoading': {
        'MaintenanceNotes': [
            "Gas rate reduction linked to suspected liquid buildup—initiated plunger lift test.",
            "Scheduled foam injection to aid unloading process.",
            "Observations indicate fluid fallback; adjusted lift parameters accordingly.",
            "Manually unloaded well following unsuccessful chemical lift.",
            "Revised choke settings to mitigate tubing pressure rise due to loading.",
            "Lift assist tools redeployed to counteract fluid accumulation trend."
        ],
        'Observations': [
            "TP rose ~150 psi while gas flow declined ~20%. Suggests classic loading behavior over extended period.",
            "Well sounds indicate presence of fallback fluid. Gas-liquid interface movement slowing down markedly.",
            "Separator reported increased slugging. Tubing head pressure fluctuating with liquid surges.",
            "Tracer analysis shows poor fluid evacuation efficiency in tubing lower section.",
            "Retro-analysis of GWR indicates gradual shift toward higher condensate ratios—consistent with loading.",
            "Flow instability pairs with rising hydrostatic pressure. Plunger lift advised before drawdown worsens."
        ]
    },
    'TubingCollapse': {
        'MaintenanceNotes': [
            "Emergency shut-in executed following sudden flow drop. Suspect tubing collapse.",
            "Logged high pressure event—downhole imaging scheduled.",
            "Pressure profile anomaly suggests mechanical tubing failure; inspection underway.",
            "Caliper logging tool deployed to verify suspected deformation.",
            "Field team dispatched for downhole diagnostics. Tubing breach likely.",
            "Prepared contingency plan for sectional tubing retrieval."
        ],
        'Observations': [
            "Flow rate fell to zero within 30 minutes. Corresponding upstream pressure increased rapidly—signaling restriction or collapse.",
            "Acoustic log shows irregular reflection near mid-tubing section. Echo pattern consistent with ovalization of pipe.",
            "Integrity test aborted due to unresponsive pressure return. Suggests sudden structural disruption in tubing path.",
            "Wellhead response shows a back-pressure pulse followed by a complete flow drop—typical for buckling or collapse.",
            "Telemetry flagged differential stress zones; prior maintenance records show thin-wall segment at failure depth.",
            "Post-event simulation indicates that pressure shock could have exceeded tubing yield strength under localized stress."
        ]
    },
    'OverHeating': {
        'MaintenanceNotes': [
            "Unit tripped on thermal alarm—cooling system flushed and restarted.",
            "High-temperature shutdown occurred. Circulating pump replaced.",
            "Lubricant level corrected after rising temp noted at gearbox.",
            "Electrical cabinet vent clogged—heat dissipation compromised.",
            "ESP inspection reveals bearing discoloration; suspect overheat stress.",
            "Adjusted setpoint temp from 95°C to 85°C following repeated thermal excursions."
        ],
        'Observations': [
            "Temperature increased at 2.8°C/hour without load change. Indicative of failing cooler or insulation breach.",
            "Thermal map shows heat concentration near drive end—cooling jacket outflow abnormally low.",
            "IR scan flagged hot spot at motor housing. Thermal resistance higher than baseline. Possible friction buildup.",
            "Current/load curve deviated ~90 mins before trip. Suggest early bearing wear due to alignment issue.",
            "Upstream valve shows delayed actuation—may have created fluid stagnation leading to heat rise.",
            "Fan differential pressure unexpectedly flat. Heat exchanger performance 30% below nominal."
        ]
    },
    'SandProduction': {
        'MaintenanceNotes': [
            "Increased sand content logged. Separator flushed and checked.",
            "Filter elements replaced after solids detection at test point.",
            "Flow instability traced to sand breakthrough. Adjusted drawdown.",
            "Activated backup separator due to screen clogging.",
            "Notified reservoir team; formation stability in question.",
            "Added erosion protection sleeve to suspected high-wear zone."
        ],
        'Observations': [
            "Sudden spike in noise and pressure fluctuation indicates particle intrusion near choke.",
            "Grain samples from return line confirmed presence of formation sand. Choke port partially obstructed.",
            "Transient drop in downstream pressure and surging signs consistent with bypassing particles.",
            "Erosion telemetry logged rapid wear progression—likely correlated to solids ingress during drawdown ramp.",
            "Wellhead vibration trending beyond 1.8x nominal amplitude—common during sanding events.",
            "Chemical profile shows rising silica levels—matches prior high-sand event from adjacent lateral."
        ]
    }
}

def add_style_modifiers(text):
    """
    Applies random stylistic modifications to log templates to simulate natural variation.

    Parameters:
    - text (str): Raw maintenance or observation string.

    Returns:
    - str: Stylized version of input string.
    """
    modifiers = [
        lambda x: x,  # no change
        lambda x: x.replace("suspected", "maybe?"),
        lambda x: x.replace("generated", "created"),
        lambda x: x.replace("generated", "developed"),
        lambda x: x.replace("FlowRate", "FR").replace("Temperature", "Temp"),
        lambda x: x.replace("FlowRate", "rate").replace("Temperature", "T"),
        lambda x: x.lower().capitalize(),
        lambda x: x + " (check reqd)",
        lambda x: x.replace("—", "-")
    ]
    text = random.choice(modifiers)(text)
    text = random.choice(modifiers)(text)
    text = random.choice(modifiers)(text)
    return text

def generate_operator_logs(logs_df, anomaly_type, start_timestamp, end_timestamp, random_seed=32):
    """
    Appends a new operator log record for a given anomaly type with stylized notes.

    Parameters:
    - logs_df (pd.DataFrame): Existing log dataframe to append to.
    - anomaly_type (str): Type of anomaly (e.g., 'TubingBlockage').
    - start_timestamp (datetime): Start time of detected anomaly.
    - end_timestamp (datetime): End time of detected anomaly.
    - random_seed (int): Seed to ensure reproducible shuffling.

    Returns:
    - pd.DataFrame: Updated log dataframe with new anomaly entry.
    """
    if random_seed is not None:
        random.seed(random_seed)

    templates = downtime_log_templates.get(anomaly_type, {'MaintenanceNotes':["Unusual behavior."], 'Observations':["Unusual behavior."]})
    maint_templates = templates.get('MaintenanceNotes')
    obs_templates = templates.get('Observations')

    random.shuffle(maint_templates)
    random.shuffle(obs_templates)
    raw_maint = maint_templates[0]
    raw_obs = obs_templates[0]
    # raw_note = random.choice(templates)
    styled_maint = add_style_modifiers(raw_maint)
    styled_obs = add_style_modifiers(raw_obs)

    logs = {
        'StartTimestamp': start_timestamp,
        'EndTimestamp': end_timestamp,
        'AnomalyType': anomaly_type,
        'MaintenanceNotes': styled_maint,
        'Observations': styled_obs,
    }

    logs_df.loc[len(logs_df)] = logs

    return logs_df
