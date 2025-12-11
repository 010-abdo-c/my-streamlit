import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import deque

# دالة FCFS المحسنة
def fcfs(processes):
    processes = sorted(processes, key=lambda x: x['arrival'])
    t = 0
    results = []
    
    for p in processes:
        if t < p['arrival']:
            t = p['arrival']
        
        start = t
        finish = t + p['burst']
        waiting = start - p['arrival']
        turnaround = finish - p['arrival']
        response = waiting
        
        results.append({
            'pid': p['pid'],
            'arrival': p['arrival'],
            'burst': p['burst'],
            'start': start,
            'finish': finish,
            'waiting': waiting,
            'turnaround': turnaround,
            'response': response
        })
        
        t = finish
    
    return results

# دالة SJF المحسنة
def sjf(processes):
    processes = sorted(processes, key=lambda x: x['arrival'])
    t = 0
    remain = processes.copy()
    results = []

    while remain:
        ready = [p for p in remain if p['arrival'] <= t]
        
        if not ready:
            t = min(remain, key=lambda x: x['arrival'])['arrival']
            continue

        p = min(ready, key=lambda x: x['burst'])
        start = t
        finish = t + p['burst']
        waiting = start - p['arrival']
        turnaround = finish - p['arrival']
        response = waiting
        
        results.append({
            'pid': p['pid'],
            'arrival': p['arrival'],
            'burst': p['burst'],
            'start': start,
            'finish': finish,
            'waiting': waiting,
            'turnaround': turnaround,
            'response': response
        })
        
        t = finish
        remain.remove(p)

    return results

# دالة Round Robin المحسنة
def round_robin(processes, quantum):
    t = 0
    queue = deque()
    remain = [{'pid': p['pid'], 'arrival': p['arrival'], 'burst': p['burst'], 'left': p['burst']} for p in processes]
    remain.sort(key=lambda x: x['arrival'])
    
    timeline = []
    start_times = {}
    finish_times = {}
    
    arrived_set = set()
    
    while remain or queue:
        # إضافة العمليات التي وصلت
        for p in list(remain):
            if p['arrival'] <= t and p['pid'] not in arrived_set:
                queue.append(p)
                arrived_set.add(p['pid'])
                remain.remove(p)
        
        if not queue:
            if remain:
                t = min(remain, key=lambda x: x['arrival'])['arrival']
                continue
            else:
                break
        
        p = queue.popleft()
        start = t
        
        # تسجيل أول مرة تبدأ فيها العملية (Response Time)
        if p['pid'] not in start_times:
            start_times[p['pid']] = start
        
        run = min(quantum, p['left'])
        t += run
        p['left'] -= run
        finish = t
        
        timeline.append({'pid': p['pid'], 'start': start, 'finish': finish})
        
        # إضافة العمليات الجديدة التي وصلت أثناء التنفيذ
        for x in list(remain):
            if x['arrival'] <= t and x['pid'] not in arrived_set:
                queue.append(x)
                arrived_set.add(x['pid'])
                remain.remove(x)
        
        if p['left'] > 0:
            queue.append(p)
        else:
            finish_times[p['pid']] = finish
    
    # حساب المقاييس لكل عملية
    results = []
    for p in processes:
        arrival = p['arrival']
        burst = p['burst']
        finish = finish_times[p['pid']]
        response = start_times[p['pid']] - arrival
        turnaround = finish - arrival
        waiting = turnaround - burst
        
        results.append({
            'pid': p['pid'],
            'arrival': arrival,
            'burst': burst,
            'start': start_times[p['pid']],
            'finish': finish,
            'waiting': waiting,
            'turnaround': turnaround,
            'response': response
        })
    
    return timeline, results

# دالة Priority Scheduling (Non-Preemptive)
def priority_scheduling(processes):
    processes = sorted(processes, key=lambda x: x['arrival'])
    t = 0
    remain = processes.copy()
    results = []

    while remain:
        ready = [p for p in remain if p['arrival'] <= t]
        
        if not ready:
            t = min(remain, key=lambda x: x['arrival'])['arrival']
            continue

        # أقل رقم أولوية = أعلى أولوية
        p = min(ready, key=lambda x: x['priority'])
        start = t
        finish = t + p['burst']
        waiting = start - p['arrival']
        turnaround = finish - p['arrival']
        response = waiting
        
        results.append({
            'pid': p['pid'],
            'arrival': p['arrival'],
            'burst': p['burst'],
            'priority': p['priority'],
            'start': start,
            'finish': finish,
            'waiting': waiting,
            'turnaround': turnaround,
            'response': response
        })
        
        t = finish
        remain.remove(p)

    return results

# حساب المقاييس
def calculate_metrics(results):
    avg_waiting = sum(r['waiting'] for r in results) / len(results)
    avg_turnaround = sum(r['turnaround'] for r in results) / len(results)
    avg_response = sum(r['response'] for r in results) / len(results)
    total_time = max(r['finish'] for r in results)
    total_burst = sum(r['burst'] for r in results)
    cpu_utilization = (total_burst / total_time * 100) if total_time > 0 else 0
    throughput = len(results) / total_time if total_time > 0 else 0
    
    return {
        'avg_waiting': avg_waiting,
        'avg_turnaround': avg_turnaround,
        'avg_response': avg_response,
        'total_time': total_time,
        'cpu_utilization': cpu_utilization,
        'throughput': throughput
    }

# رسم Gantt Chart محسن
def draw_gantt(timeline, title, is_rr=False):
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
    
    for idx, row in enumerate(timeline):
        pid = row['pid']
        duration = row['finish'] - row['start']
        
        fig.add_trace(go.Bar(
            x=[duration],
            y=['CPU'],
            orientation='h',
            base=row['start'],
            marker=dict(
                color=colors[(pid-1) % len(colors)],
                line=dict(color='white', width=1)
            ),
            text=f"P{pid}",
            textposition='inside',
            textfont=dict(size=10, color='white', family='Arial Black'),
            name=f"P{pid}",
            showlegend=False,
            hovertemplate=f"<b>Process {pid}</b><br>Start: {row['start']}<br>End: {row['finish']}<br>Duration: {duration}<extra></extra>"
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="الوقت (Time Units)",
        yaxis_title="",
        height=250,
        barmode='stack',
        showlegend=False,
        template='plotly_white',
        xaxis=dict(dtick=1, showgrid=True, gridcolor='lightgray'),
        plot_bgcolor='rgba(240,240,240,0.5)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# جدول تفصيلي للنتائج
def show_detailed_table(results, algo_name):
    df = pd.DataFrame(results)
    df = df.sort_values('pid')
    
    display_df = pd.DataFrame({
        'Process': [f"P{r['pid']}" for r in results],
        'Arrival': [r['arrival'] for r in results],
        'Burst': [r['burst'] for r in results],
        'Start': [r['start'] for r in results],
        'Finish': [r['finish'] for r in results],
        'Waiting': [r['waiting'] for r in results],
        'Turnaround': [r['turnaround'] for r in results],
        'Response': [r['response'] for r in results]
    })
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# واجهة التطبيق
st.set_page_config(page_title="CPU Scheduler Pro", layout="wide", page_icon="🖥️")

st.title("🖥️ محاكي جدولة المعالج المحسن - Advanced CPU Scheduling Simulator")
st.markdown("*نسخة محسنة مع حسابات دقيقة ومقاييس متقدمة*")
st.markdown("---")

# Sidebar للتحكم
with st.sidebar:
    st.header("⚙️ الإعدادات")
    quantum = st.slider("Round Robin Quantum", min_value=1, max_value=10, value=2, step=1)
    
    st.markdown("---")
    st.subheader("📖 الشرح")
    st.markdown("""
    **المقاييس:**
    - **Waiting Time**: الوقت في الانتظار
    - **Turnaround Time**: الوقت الكلي
    - **Response Time**: وقت الاستجابة الأول
    - **CPU Utilization**: نسبة استخدام المعالج
    - **Throughput**: عدد العمليات/وحدة زمن
    """)

# إنشاء جدول البيانات
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame({
        'Process': ['P1', 'P2', 'P3', 'P4'],
        'Arrival Time': [0, 1, 2, 3],
        'Burst Time': [5, 3, 8, 6],
        'Priority': [2, 1, 4, 3]
    })

st.subheader("📊 جدول العمليات - Process Table")

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 إعادة تعيين"):
        st.session_state.df = pd.DataFrame({
            'Process': ['P1', 'P2', 'P3', 'P4'],
            'Arrival Time': [0, 1, 2, 3],
            'Burst Time': [5, 3, 8, 6],
            'Priority': [2, 1, 4, 3]
        })
        st.rerun()

edited_df = st.data_editor(
    st.session_state.df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "Process": st.column_config.TextColumn("العملية", width="small"),
        "Arrival Time": st.column_config.NumberColumn("وقت الوصول", min_value=0, max_value=100, step=1),
        "Burst Time": st.column_config.NumberColumn("وقت التنفيذ", min_value=1, max_value=100, step=1),
        "Priority": st.column_config.NumberColumn("الأولوية", min_value=1, max_value=10, step=1)
    }
)

st.session_state.df = edited_df

st.markdown("---")

# تحويل البيانات
processes = []
for idx, row in edited_df.iterrows():
    processes.append({
        'pid': idx + 1,
        'arrival': int(row['Arrival Time']),
        'burst': int(row['Burst Time']),
        'priority': int(row['Priority'])
    })

# حساب النتائج
fcfs_results = fcfs(processes)
sjf_results = sjf(processes)
rr_timeline, rr_results = round_robin(processes, quantum)
priority_results = priority_scheduling(processes)

# حساب المقاييس
fcfs_metrics = calculate_metrics(fcfs_results)
sjf_metrics = calculate_metrics(sjf_results)
rr_metrics = calculate_metrics(rr_results)
priority_metrics = calculate_metrics(priority_results)

# التبويبات
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 FCFS", "📊 SJF", "🔄 Round Robin", "⭐ Priority", "📉 المقارنة الشاملة", "📋 الجداول التفصيلية"
])

with tab1:
    st.subheader("First Come First Serve (FCFS)")
    timeline = [{'pid': r['pid'], 'start': r['start'], 'finish': r['finish']} for r in fcfs_results]
    draw_gantt(timeline, "FCFS Gantt Chart")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("متوسط الانتظار", f"{fcfs_metrics['avg_waiting']:.2f}")
    with col2:
        st.metric("متوسط Turnaround", f"{fcfs_metrics['avg_turnaround']:.2f}")
    with col3:
        st.metric("متوسط Response", f"{fcfs_metrics['avg_response']:.2f}")
    with col4:
        st.metric("CPU Utilization", f"{fcfs_metrics['cpu_utilization']:.1f}%")
    with col5:
        st.metric("Throughput", f"{fcfs_metrics['throughput']:.3f}")
    
    show_detailed_table(fcfs_results, "FCFS")

with tab2:
    st.subheader("Shortest Job First (SJF)")
    timeline = [{'pid': r['pid'], 'start': r['start'], 'finish': r['finish']} for r in sjf_results]
    draw_gantt(timeline, "SJF Gantt Chart")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("متوسط الانتظار", f"{sjf_metrics['avg_waiting']:.2f}")
    with col2:
        st.metric("متوسط Turnaround", f"{sjf_metrics['avg_turnaround']:.2f}")
    with col3:
        st.metric("متوسط Response", f"{sjf_metrics['avg_response']:.2f}")
    with col4:
        st.metric("CPU Utilization", f"{sjf_metrics['cpu_utilization']:.1f}%")
    with col5:
        st.metric("Throughput", f"{sjf_metrics['throughput']:.3f}")
    
    show_detailed_table(sjf_results, "SJF")

with tab3:
    st.subheader(f"Round Robin (Quantum = {quantum})")
    draw_gantt(rr_timeline, f"Round Robin Gantt Chart (Q={quantum})", is_rr=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("متوسط الانتظار", f"{rr_metrics['avg_waiting']:.2f}")
    with col2:
        st.metric("متوسط Turnaround", f"{rr_metrics['avg_turnaround']:.2f}")
    with col3:
        st.metric("متوسط Response", f"{rr_metrics['avg_response']:.2f}")
    with col4:
        st.metric("CPU Utilization", f"{rr_metrics['cpu_utilization']:.1f}%")
    with col5:
        st.metric("Throughput", f"{rr_metrics['throughput']:.3f}")
    
    show_detailed_table(rr_results, "Round Robin")

with tab4:
    st.subheader("Priority Scheduling (Non-Preemptive)")
    timeline = [{'pid': r['pid'], 'start': r['start'], 'finish': r['finish']} for r in priority_results]
    draw_gantt(timeline, "Priority Scheduling Gantt Chart")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("متوسط الانتظار", f"{priority_metrics['avg_waiting']:.2f}")
    with col2:
        st.metric("متوسط Turnaround", f"{priority_metrics['avg_turnaround']:.2f}")
    with col3:
        st.metric("متوسط Response", f"{priority_metrics['avg_response']:.2f}")
    with col4:
        st.metric("CPU Utilization", f"{priority_metrics['cpu_utilization']:.1f}%")
    with col5:
        st.metric("Throughput", f"{priority_metrics['throughput']:.3f}")
    
    show_detailed_table(priority_results, "Priority")

with tab5:
    st.subheader("📊 المقارنة الشاملة بين الخوارزميات")
    
    comparison_df = pd.DataFrame({
        'الخوارزمية': ['FCFS', 'SJF', 'Round Robin', 'Priority'],
        'Waiting Time': [fcfs_metrics['avg_waiting'], sjf_metrics['avg_waiting'], 
                        rr_metrics['avg_waiting'], priority_metrics['avg_waiting']],
        'Turnaround Time': [fcfs_metrics['avg_turnaround'], sjf_metrics['avg_turnaround'],
                           rr_metrics['avg_turnaround'], priority_metrics['avg_turnaround']],
        'Response Time': [fcfs_metrics['avg_response'], sjf_metrics['avg_response'],
                         rr_metrics['avg_response'], priority_metrics['avg_response']],
        'CPU Utilization': [fcfs_metrics['cpu_utilization'], sjf_metrics['cpu_utilization'],
                           rr_metrics['cpu_utilization'], priority_metrics['cpu_utilization']],
        'Throughput': [fcfs_metrics['throughput'], sjf_metrics['throughput'],
                      rr_metrics['throughput'], priority_metrics['throughput']]
    })
    
    # رسم المقارنات
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(comparison_df, x='الخوارزمية', y='Waiting Time',
                     title='مقارنة متوسط وقت الانتظار',
                     color='الخوارزمية',
                     color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        fig1.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        fig1.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(comparison_df, x='الخوارزمية', y='Turnaround Time',
                     title='مقارنة متوسط Turnaround Time',
                     color='الخوارزمية',
                     color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        fig2.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        fig2.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig2, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig3 = px.bar(comparison_df, x='الخوارزمية', y='Response Time',
                     title='مقارنة متوسط Response Time',
                     color='الخوارزمية',
                     color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        fig3.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        fig3.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        fig4 = px.line(comparison_df, x='الخوارزمية', y='CPU Utilization',
                      title='مقارنة استخدام المعالج',
                      markers=True)
        fig4.update_traces(texttemplate='%{y:.1f}%', textposition='top center')
        fig4.update_layout(height=350)
        st.plotly_chart(fig4, use_container_width=True)
    
    st.dataframe(comparison_df.style.highlight_min(subset=['Waiting Time', 'Turnaround Time', 'Response Time'], color='lightgreen')
                                    .highlight_max(subset=['CPU Utilization', 'Throughput'], color='lightgreen'),
                use_container_width=True, hide_index=True)
    
    # أفضل خوارزمية
    best_waiting = comparison_df.loc[comparison_df['Waiting Time'].idxmin(), 'الخوارزمية']
    best_turnaround = comparison_df.loc[comparison_df['Turnaround Time'].idxmin(), 'الخوارزمية']
    best_response = comparison_df.loc[comparison_df['Response Time'].idxmin(), 'الخوارزمية']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"🏆 أقل Waiting Time: **{best_waiting}**")
    with col2:
        st.success(f"🏆 أقل Turnaround Time: **{best_turnaround}**")
    with col3:
        st.success(f"🏆 أقل Response Time: **{best_response}**")

with tab6:
    st.subheader("📋 الجداول التفصيلية لجميع الخوارزميات")
    
    st.markdown("### FCFS")
    show_detailed_table(fcfs_results, "FCFS")
    
    st.markdown("### SJF")
    show_detailed_table(sjf_results, "SJF")
    
    st.markdown("### Round Robin")
    show_detailed_table(rr_results, "Round Robin")
    
    st.markdown("### Priority")
    show_detailed_table(priority_results, "Priority")
